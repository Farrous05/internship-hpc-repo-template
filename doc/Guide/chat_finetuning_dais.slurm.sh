#!/bin/bash -l

#SBATCH -o .log/job.out
#SBATCH -e .log/job.err
#SBATCH -D ./
#SBATCH -J chat-finetuning

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB

#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1

# Wall clock limit (max. is 24 hours):
#SBATCH --time=01:00:00

module purge
module load apptainer

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "[$(date)] Loading environment variables from .env file..."
    set -a
    source .env
    set +a
else
    echo "[$(date)] Warning: .env file not found"
fi

echo "[$(date)] Starting apptainer instance..."
srun apptainer exec \
  --nv \
  --bind $PWD:/workspace \
  --bind "${HF_HOME}":"/root/.cache/huggingface" \
  --env HF_HOME="/root/.cache/huggingface" \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env WANDB_API_KEY="${WANDB_API_KEY}" \
  --env WANDB_PROJECT="${WANDB_PROJECT}" \
  container/llm-dev-env.sif \
  python scripts/fine_tuning/chat_finetuning.py