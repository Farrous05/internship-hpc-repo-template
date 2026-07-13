"""General-purpose ModelAssistant for loading, training, and inference.

This module provides a flexible ModelAssistant class that can work with
any Hugging Face transformer model, with defaults optimized for GPT-2.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
)

from config import ModelAssistantConfig, TrainingConfig

logger = logging.getLogger(__name__)


class ModelAssistant:
    """General-purpose assistant for model operations.

    Supports loading, training, and inference with Hugging Face models.
    Defaults to GPT-2 for simplicity and broad compatibility.
    """

    def __init__(self, config: ModelAssistantConfig):
        """Initialize the ModelAssistant.

        Args:
            config: Configuration for the model assistant
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer from the configured path."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path
            )
        except Exception as e:
            logger.warning(
                f"Could not load tokenizer from {self.config.model_path}: {e}"
            )
            logger.info("Falling back to GPT-2 tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path
            )
        except Exception as e:
            logger.error(
                f"Could not load model from {self.config.model_path}: {e}"
            )
            raise

        # Ensure pad_token is set (required for training)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Auto-detect device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        logger.info(f"Model loaded on device: {self.device}")

    def generate_text(
        self, prompt: str, generation_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate text from a given prompt.

        Args:
            prompt: Input text to generate from
            generation_params: Optional parameters for generation

        Returns:
            Generated text (excluding the original prompt)
        """
        if generation_params is None:
            generation_params = self.config.generation_params

        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        n_input = input_ids.shape[1]

        # Generate
        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(input_ids, **generation_params)

        # Decode only new tokens
        generated_text = self.tokenizer.decode(
            output[0, n_input:], skip_special_tokens=True
        )

        return generated_text

    @staticmethod
    def load_dataset_from_path(dataset_path: str) -> Dataset:
        """Load a dataset from various file formats.

        Args:
            dataset_path: Path to the dataset file

        Returns:
            Dataset object with 'text' field

        Raises:
            ValueError: If dataset format is unsupported or loading fails
        """
        dataset_path = Path(dataset_path)

        if dataset_path.suffix == ".txt":
            # Load text file
            with open(dataset_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
            return Dataset.from_dict({"text": texts})
        else:
            # Try loading as HF dataset
            try:
                return load_dataset(str(dataset_path))["train"]
            except Exception as e:
                raise ValueError(
                    f"Could not load dataset from {dataset_path}: {e}"
                )

    def prepare_dataset_for_training(
        self,
        dataset_path: str,
        max_length: int = 512,
        train_test_split: float = 0.8,
    ) -> Dict[str, Dataset]:
        """Prepare a dataset for training.

        Args:
            dataset_path: Path to the dataset
            max_length: Maximum sequence length
            train_test_split: Fraction for training (rest for eval)

        Returns:
            Dictionary with 'train' and 'eval' datasets
        """
        # Load raw dataset
        dataset = self.load_dataset_from_path(dataset_path)

        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        # Split into train/eval
        split_dataset = tokenized_dataset.train_test_split(
            test_size=1 - train_test_split
        )

        return {"train": split_dataset["train"], "eval": split_dataset["test"]}

    def train(self, config: TrainingConfig):
        """Train the model using the provided configuration.

        Args:
            config: Training configuration
        """
        logger.info("Starting model training...")

        # Prepare dataset
        datasets = self.prepare_dataset_for_training(
            config.dataset_path, config.max_length, config.train_test_split
        )

        # Use training arguments from config
        training_args = config.training_args

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False  # Causal LM, not masked LM
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["eval"],
            data_collator=data_collator,
            processing_class=self.tokenizer,
        )

        # Train
        trainer.train()

        # Save the model
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_path))

        # Save config metadata
        with open(output_path / "training_config.json", "w") as f:
            json.dump(config.model_dump(), f, indent=2)

        logger.info(f"Training completed. Model saved to {output_path}")
