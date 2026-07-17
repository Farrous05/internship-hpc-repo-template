import argparse
import os
import sys

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.utils.load_model import load_model_and_tokenizer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_DATASET = "tatsu-lab/alpaca"
DEFAULT_OUTPUT_DIR = "models/chat-finetuned"


def parse_args():
    parser = argparse.ArgumentParser(description="Chat fine-tuning with LoRA")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def format_example(example, tokenizer):
    messages = [{"role": "user", "content": example["instruction"]}]
    if example.get("input"):
        messages[0]["content"] += f"\n\n{example['input']}"
    messages.append({"role": "assistant", "content": example["output"]})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    load_dotenv()
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model, use_lora=True)

    raw = load_dataset(args.dataset, split="train")
    # Keep only examples with non-empty output
    raw = raw.filter(lambda x: x["output"] and x["output"].strip())

    def tokenize(example):
        text = format_example(example, tokenizer)
        encoded = tokenizer(text, truncation=True, max_length=args.max_length, padding=False)
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    dataset = raw.map(tokenize, remove_columns=raw.column_names)
    split = dataset.train_test_split(test_size=0.05, seed=42)

    report_to = "wandb" if os.getenv("WANDB_API_KEY") else "none"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        run_name="chat-finetuning",
        report_to=report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    # Save only the LoRA adapter weights, not the full model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    print("HF_HOME:", os.getenv("HF_HOME"))
    main()