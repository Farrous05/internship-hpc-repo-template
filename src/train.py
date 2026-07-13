#!/usr/bin/env python3
"""Training script for ModelAssistant with Hugging Face Transformers.

This script provides a clean interface for training language models
using the ModelAssistant class and YAML configuration files.
Compatible with SLURM job submission via run_slurm.py.
"""

import argparse
import logging
import sys
from pathlib import Path

from config import TrainingConfig
from model_assistant import ModelAssistant
from utils import load_config_from_yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Train a language model using ModelAssistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with YAML config
  python -m src.train --config config.yaml

  # Train with specific output directory
  python -m src.train --config config.yaml --output_dir ./my_model

  # Training with SLURM (via run_slurm.py)
  python src/run_slurm.py config.yaml --script src/train.py
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--output_dir", type=str, help="Override output directory from config"
    )

    parser.add_argument(
        "--dataset_path", type=str, help="Override dataset path from config"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configuration without starting training",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser


def load_and_process_config(args: argparse.Namespace) -> TrainingConfig:
    """Load configuration from YAML and apply CLI overrides.

    Args:
        args: Parsed command-line arguments

    Returns:
        Processed TrainingConfig instance
    """
    # Load configuration from YAML
    config = load_config_from_yaml(args.config, TrainingConfig)
    logger.info(f"Loaded configuration from {args.config}")

    # Apply CLI overrides
    if args.output_dir:
        config.output_dir = args.output_dir
        config.training_args["output_dir"] = args.output_dir

    if args.dataset_path:
        config.dataset_path = args.dataset_path

    return config


def validate_training_setup(config: TrainingConfig) -> None:
    """Validate training setup requirements.

    Args:
        config: Training configuration to validate

    Raises:
        FileNotFoundError: If dataset path doesn't exist
    """
    # Check dataset path
    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Check output directory
    output_dir = Path(config.output_dir)
    # overwrite_output_dir was removed from TrainingArguments in
    # transformers 5.x; default to True (allow overwrite) when absent.
    overwrite_allowed = getattr(
        config.training_args, "overwrite_output_dir", True
    )

    if output_dir.exists() and not overwrite_allowed:
        logger.warning(
            f"Output directory {output_dir} exists. "
            "Set overwrite_output_dir=True to overwrite."
        )

    logger.info("Training setup validation passed")


def execute_training_workflow(config: TrainingConfig) -> None:
    """Execute the complete training workflow.

    Args:
        config: Training configuration
    """
    logger.info("Initializing ModelAssistant...")
    assistant = ModelAssistant(config.assistant_config)

    logger.info("Starting training process...")
    assistant.train(config)

    logger.info("Training completed successfully!")


def main() -> None:
    """Main entry point for the training script."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    try:
        # Load and process configuration
        config = load_and_process_config(args)

        # Validate setup
        validate_training_setup(config)

        # Handle dry run
        if args.dry_run:
            logger.info(
                "Dry run completed successfully. Configuration is valid."
            )
            return

        # Execute training
        execute_training_workflow(config)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            logger.exception("Full error traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
