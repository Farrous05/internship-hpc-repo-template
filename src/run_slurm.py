#!/usr/bin/env python3
"""
SLURM Job Submission Tool.

A comprehensive tool for submitting jobs to SLURM with YAML configuration
support, placeholder replacement, and resource management.
"""

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

# Constants
DEFAULT_TEMPLATE_PATH = "src/slurm_scripts/template.slurm"
DEFAULT_SCRIPT_PATH = "src/train.py"

# Default values for job submission
# PLEASE CHANGE THESE VALUES TO YOUR OWN PROJECT NAME AND JOB NAME
DEFAULT_PROJECT_NAME = "hpc_repo_template"
DEFAULT_JOB_NAME = "v1"
DEFAULT_GROUP_NAME = "hpc_demo"
DEFAULT_ENTITY_NAME = "faresshretah-max-planck-society"

DEFAULT_IMAGE_PATH = (
    "/u/fash/internship-hpc-repo-template/container/llm-dev-env.sif"
)
DEFAULT_TIME = "00:10:00"

# Resource calculation constants
GPUS_PER_NODE = 4
CORES_PER_GPU = 12
MEMORY_PER_GPU = 125000


class SlurmJobError(Exception):
    """Custom exception for SLURM job submission errors."""

    pass


class ConfigProcessor:
    """Handles YAML configuration processing and placeholder replacement."""

    @staticmethod
    def generate_job_id() -> str:
        """Generate a unique job ID based on current timestamp."""
        return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

    @staticmethod
    def parse_dynamic_args(
        unknown_args: List[str], argv: List[str]
    ) -> Dict[str, str]:
        """Parse unknown command line arguments into a dictionary."""
        dynamic_args = {}
        for arg in unknown_args:
            if arg.startswith("--"):
                key = arg.lstrip("-")
                try:
                    idx = argv.index(arg)
                    if idx + 1 < len(argv):
                        dynamic_args[key] = argv[idx + 1]
                except ValueError:
                    continue
        return dynamic_args

    def load_and_merge_configs(
        self, config_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Load main config and merge with included configurations."""
        try:
            with open(config_path, "r") as file:
                main_config = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise SlurmJobError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise SlurmJobError(f"Invalid YAML in {config_path}: {e}")

        # Process includes
        includes = self._find_include_value(main_config, "__include")
        if includes:
            for include_path in includes:
                included_config = self._load_include_config(include_path)
                self._deep_merge_configs(main_config, included_config)

        return main_config

    def replace_placeholders(
        self, config: Dict[str, Any], replacements: Dict[str, str]
    ) -> Dict[str, Any]:
        """Replace placeholders in configuration with actual values."""
        return self._replace_placeholder_recursive(config, replacements)

    def _load_include_config(
        self, include_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Load an included configuration file."""
        try:
            with open(include_path, "r") as file:
                return yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise SlurmJobError(f"Include file not found: {include_path}")
        except yaml.YAMLError as e:
            raise SlurmJobError(f"Invalid YAML in {include_path}: {e}")

    def _deep_merge_configs(
        self, main_config: Dict[str, Any], included_config: Dict[str, Any]
    ) -> None:
        """Deep merge included configuration into main configuration."""
        for key, value in included_config.items():
            if (
                key in main_config
                and isinstance(main_config[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge_configs(main_config[key], value)
            else:
                main_config[key] = value

    def _find_include_value(
        self, data: Any, target_key: str
    ) -> Optional[List[str]]:
        """Recursively find include directives in configuration."""
        if isinstance(data, dict):
            for key, value in data.items():
                if key == target_key:
                    return value
                elif isinstance(value, dict):
                    result = self._find_include_value(value, target_key)
                    if result is not None:
                        return result
        return None

    def _replace_placeholder_recursive(
        self, element: Any, replacements: Dict[str, str]
    ) -> Any:
        """Recursively replace placeholders in nested structures."""
        if isinstance(element, dict):
            return {
                key: self._replace_placeholder_recursive(value, replacements)
                for key, value in element.items()
            }
        elif isinstance(element, list):
            return [
                self._replace_placeholder_recursive(item, replacements)
                for item in element
            ]
        elif isinstance(element, str):
            return self._process_string_placeholder(element, replacements)
        return element

    def _process_string_placeholder(
        self, text: str, replacements: Dict[str, str]
    ) -> Any:
        """Process placeholder in string and convert to appropriate type."""
        pattern = r"<<(?:(\w+): )?(\w+)>>"
        matches = list(re.finditer(pattern, text))

        if len(matches) == 1 and matches[0].span() == (0, len(text)):
            return self._convert_placeholder_type(matches[0], replacements)

        # Multiple placeholders or additional text
        def string_replacement(match: re.Match) -> str:
            result = self._convert_placeholder_type(match, replacements)
            return str(result)

        return re.sub(pattern, string_replacement, text)

    def _convert_placeholder_type(
        self, match: re.Match, replacements: Dict[str, str]
    ) -> Any:
        """Convert placeholder to specified type."""
        type_hint = match.group(1) if match.group(1) else "str"
        variable_name = match.group(2)
        replacement_value = replacements.get(variable_name, match.group(0))

        if replacement_value == match.group(0):  # No replacement found
            return replacement_value

        # Type conversion mapping - functional approach
        type_converters = {
            "float": lambda x: float(x),
            "int": lambda x: int(x),
            "bool": lambda x: x.lower() == "true",
            "list_int": lambda x: [int(i.strip()) for i in x.split(",")],
            "str": lambda x: x,  # Default converter
        }

        converter = type_converters.get(type_hint, type_converters["str"])

        try:
            return converter(replacement_value)
        except (ValueError, AttributeError, TypeError):
            return None


class ResourceCalculator:
    """Calculates SLURM resource requirements based on GPU count."""

    @staticmethod
    def calculate_resources(n_gpu: int) -> Dict[str, Union[int, str]]:
        """Calculate compute resources based on GPU requirements."""
        if n_gpu > GPUS_PER_NODE:
            if n_gpu % GPUS_PER_NODE != 0:
                raise SlurmJobError(
                    f"GPU count must be divisible by {GPUS_PER_NODE} "
                    f"for multi-node jobs"
                )
            n_nodes = n_gpu // GPUS_PER_NODE
            actual_gpus = GPUS_PER_NODE
        else:
            n_nodes = 1
            actual_gpus = n_gpu

        memory = (
            0 if actual_gpus >= GPUS_PER_NODE else MEMORY_PER_GPU * actual_gpus
        )
        cpu = actual_gpus * CORES_PER_GPU

        return {
            "n_nodes": n_nodes,
            "n_gpu": actual_gpus,
            "n_cpu": cpu,
            "partition": "gpu",
            "memory": memory,
        }


class FileManager:
    """Manages file operations for job submission."""

    @staticmethod
    def ensure_output_directory(
        output_dir: Optional[str], group_name: str, job_name: str, job_id: str
    ) -> Path:
        """Create and return the output directory path."""
        if output_dir is None:
            output_path = Path("experiments") / group_name / job_name / job_id
        else:
            output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    @staticmethod
    def copy_config_to_output(
        config: Dict[str, Any], output_dir: Path, job_id: str
    ) -> Path:
        """Copy processed configuration to job-specific directory."""
        dest_filename = f"{job_id}.yml"
        dest_path = output_dir / dest_filename

        with open(dest_path, "w") as file:
            yaml.dump(config, file, sort_keys=False)

        return dest_path

    @staticmethod
    def generate_slurm_script(
        template_path: Union[str, Path],
        output_dir: Path,
        job_id: str,
        script_args: Dict[str, Any],
    ) -> Path:
        """Generate SLURM script from template."""
        try:
            with open(template_path, "r") as file:
                script_content = file.read().format(**script_args)
        except FileNotFoundError:
            raise SlurmJobError(f"Template file not found: {template_path}")
        except KeyError as e:
            raise SlurmJobError(
                f"Missing template variable in {template_path}: {e}"
            )

        output_script = output_dir / f"{job_id}.sh"
        with open(output_script, "w") as file:
            file.write(script_content)

        return output_script


class SlurmSubmitter:
    """Handles SLURM job submission."""

    @staticmethod
    def submit_job(script_path: Path, dry_run: bool = False) -> None:
        """Submit job to SLURM or perform dry run."""
        if dry_run:
            print(f"Generated script at {script_path} without submission.")
        else:
            try:
                result = subprocess.run(
                    ["sbatch", str(script_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(f"Job submitted successfully: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                raise SlurmJobError(f"Failed to submit job: {e.stderr}")
            except FileNotFoundError:
                raise SlurmJobError(
                    "sbatch command not found. Is SLURM installed?"
                )


class SlurmJobManager:
    """Main class orchestrating SLURM job submission workflow."""

    def __init__(self):
        self.config_processor = ConfigProcessor()
        self.resource_calculator = ResourceCalculator()
        self.file_manager = FileManager()
        self.submitter = SlurmSubmitter()

    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Create and configure command line argument parser."""
        parser = argparse.ArgumentParser(
            description="Submit SLURM jobs with YAML configuration support."
        )

        # Argument configuration mapping
        argument_specs = {
            # Required arguments
            "config_file": {
                "type": str,
                "required": True,
                "help": "Path to the YAML configuration file.",
            },
            # Optional file/path arguments
            "output_dir": {
                "type": str,
                "default": None,
                "help": "Output directory path. Auto-generated if not specified.",  # noqa: E501
            },
            "template": {
                "type": str,
                "default": DEFAULT_TEMPLATE_PATH,
                "help": f"Path to SLURM script template (default: {DEFAULT_TEMPLATE_PATH})",  # noqa: E501
            },
            "script": {
                "type": str,
                "default": DEFAULT_SCRIPT_PATH,
                "help": f"Script to run (default: {DEFAULT_SCRIPT_PATH})",
            },
            "image": {
                "type": str,
                "default": DEFAULT_IMAGE_PATH,
                "help": "Apptainer image to use",
            },
            # Boolean flags
            "dry": {
                "action": "store_true",
                "help": "Generate files without submitting job",
            },
            # Resource arguments
            "n_gpu": {
                "type": int,
                "default": 1,
                "help": "Number of GPUs to use (default: 1)",
            },
            "time": {
                "type": str,
                "default": DEFAULT_TIME,
                "help": f"Runtime in HH:MM:SS format (default: {DEFAULT_TIME})",  # noqa: E501
            },
            # Job metadata
            "job_name": {
                "type": str,
                "default": DEFAULT_JOB_NAME,
                "help": f"Job name (default: {DEFAULT_JOB_NAME})",
            },
            "group_name": {
                "type": str,
                "default": DEFAULT_GROUP_NAME,
                "help": f"Experiment group name (default: {DEFAULT_GROUP_NAME})",  # noqa: E501
            },
            "project_name": {
                "type": str,
                "default": DEFAULT_PROJECT_NAME,
                "help": f"Project to charge (default: {DEFAULT_PROJECT_NAME})",
            },
            "entity_name": {
                "type": str,
                "default": DEFAULT_ENTITY_NAME,
                "help": f"W&B entity/team to log runs under (default: {DEFAULT_ENTITY_NAME})",  # noqa: E501
            },
        }

        # Use functional approach to add all arguments
        for arg_name, spec in argument_specs.items():
            parser.add_argument(f"--{arg_name}", **spec)

        return parser

    def process_arguments(
        self, argv: List[str]
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Process command line arguments and
        return parsed args and dynamic args."""
        parser = self.create_argument_parser()
        known_args, unknown_args = parser.parse_known_args(argv[1:])

        args_dict = vars(known_args)
        dynamic_args = self.config_processor.parse_dynamic_args(
            unknown_args, argv
        )
        args_dict.update(dynamic_args)

        return args_dict, dynamic_args

    def run(self, argv: List[str]) -> None:
        """Main execution method."""
        try:
            # Process arguments
            args_dict, dynamic_args = self.process_arguments(argv)

            # Generate job ID and calculate resources
            job_id = self.config_processor.generate_job_id()
            resources = self.resource_calculator.calculate_resources(
                args_dict["n_gpu"]
            )
            args_dict.update(resources)
            args_dict["job_id"] = job_id

            # Setup output directory
            output_dir = self.file_manager.ensure_output_directory(
                args_dict["output_dir"],
                args_dict["group_name"],
                args_dict["job_name"],
                job_id,
            )
            args_dict["output_dir"] = str(output_dir)

            # Process configuration
            config = self.config_processor.load_and_merge_configs(
                args_dict["config_file"]
            )
            config = self.config_processor.replace_placeholders(
                config, args_dict
            )

            # Copy config to output directory
            copied_config_path = self.file_manager.copy_config_to_output(
                config, output_dir, job_id
            )
            args_dict["config_file"] = str(copied_config_path)
            args_dict["copied_config_file"] = str(copied_config_path)

            # Generate SLURM script
            script_path = self.file_manager.generate_slurm_script(
                args_dict["template"], output_dir, job_id, args_dict
            )

            # Submit job
            self.submitter.submit_job(script_path, args_dict["dry"])

        except SlurmJobError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    """Entry point for the SLURM job submission tool."""
    job_manager = SlurmJobManager()
    job_manager.run(sys.argv)


if __name__ == "__main__":
    main()
