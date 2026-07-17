import os 
from typing import Optional
import argparse
from dotenv import load_dotenv  


from datasets import load_dataset


# load env var (you should set HF_TOKEN and HF_HOME in your environment)
load_dotenv()


def download_dataset(path: str, name: Optional[str] = None):
    """Download a dataset from Hugging Face and cache it locally."""
    # Load the dataset
    if name:
        ds = load_dataset(path, name)
    else:
        ds = load_dataset(path)
    print(f"Dataset '{path}' downloaded successfully.")
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a dataset from Hugging Face")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the dataset to download from Hugging Face. Can be repo name or local path.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="Specific configuration name of the dataset (if applicable)",
    )


    args = parser.parse_args()
    path = args.path
    name = args.name
    # Load the dataset
    download_dataset(path, name)