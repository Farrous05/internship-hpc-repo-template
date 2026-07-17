#!/bin/bash -l

#SBATCH -o .log/train_end.out
#SBATCH -e .log/train_end.err
#SBATCH -D ./
#SBATCH -J train-end-token

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=125GB

#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1

# Wall clock limit (max 24h):
#SBATCH --time=06:00:00

# Method A: LoRA SFT to teach the <end> token, on the llm-dev-env image.
# Prereqs: built container/llm-dev-env.sif; the offline dataset JSONL present;
# gemma weights pre-staged in HF_HOME (login node) if the compute node is offline.

module purge
module load apptainer

if [ -f .env ]; then set -a; source .env; set +a; fi
mkdir -p .log

srun apptainer exec \
    --nv \
    --bind "$PWD":/workspace \
    --bind "${HF_HOME}":/root/.cache/huggingface \
    --env HF_HOME=/root/.cache/huggingface \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env WANDB_PROJECT="${WANDB_PROJECT:-end-token}" \
    container/llm-dev-env.sif \
    python scripts/fine_tuning/train_end_token.py \
        --model google/gemma-4-31B-it \
        `# HF id (transformers), NOT the Ollama tag -- Ollama is only for harvest` \
        --data data/offline_dataset.jsonl \
        --eval-data data/offline_dataset_heldout.jsonl \
        --output-dir models/end-token-A \
        --lora-r 16 --lora-alpha 32 --epochs 2 --max-length 4096
