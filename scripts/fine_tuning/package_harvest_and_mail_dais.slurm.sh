#!/bin/bash -l

#SBATCH -o .log/package_harvest.out
#SBATCH -e .log/package_harvest.err
#SBATCH -D /u/fash/internship-hpc-repo-template
#SBATCH -J package-harvest

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=125GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00

# Thin wrapper: builds the compact harvest artifacts and emails them. All logic
# is in scripts/package_harvest.py. Submit AFTER the (6 parallel) harvest jobs
# with a dependency so it fires once when they end (afterany => runs whether
# they COMPLETED or were killed at wall-time):
#   sbatch --dependency=afterany:<j0>:<j1>:<j2>:<j3>:<j4>:<j5> \
#          scripts/fine_tuning/package_harvest_and_mail_dais.slurm.sh

module purge
if [ -f .env ]; then set -a; source .env; set +a; fi
export HF_HOME="${HF_HOME:-/u/fash/.cache/huggingface}"
mkdir -p .log "$HF_HOME"

HARVEST_DIR="${HARVEST_DIR:-/dais/fs/scratch/$USER/results/qwen10k}"
TOPICS="${TOPICS:-/u/fash/babel-ai/data/topics_10k.json}"
SHAREGPT="${SHAREGPT:-/u/fash/babel-ai/data/sharegpt_real.json}"
NEG_POOL="${NEG_POOL:-data/healthy_generated.json}"
MAILTO="${MAILTO:-faresshretah@gmail.com}"

# Use the venv, NOT the container: llm-dev-env.sif lacks sentence_transformers
# (which the builder's bge-large embedder needs). The venv has ST + torch, and
# running on the host (no container) also lets smtplib reach the host mail relay
# on localhost:25 for the email step. bge-large uses the GPU if the venv torch
# sees CUDA, else CPU (the build works on CPU too, just slower).
source .venv/bin/activate
srun python scripts/package_harvest.py \
    --harvest-dir "$HARVEST_DIR" \
    --topics "$TOPICS" \
    --sharegpt "$SHAREGPT" \
    --neg-pool "$NEG_POOL" \
    --mailto "$MAILTO"
