#!/bin/bash -l

#SBATCH -o .log/train_qwen3b_full.out
#SBATCH -e .log/train_qwen3b_full.err
#SBATCH -D ./
#SBATCH -J train-end-qwen3b-full

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=125GB

#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1

# Wall clock (max 24h). Full FT of a 3B on 655 short examples is quick (~30 min):
#SBATCH --time=04:00:00

# Method A PoC: FULL fine-tune of Qwen2.5-3B to teach the <end> token, on the
# llm-dev-env image. Compute nodes have internet (tested 2026-07-15), so the 3B
# weights download in-job into HF_HOME; no login-node pre-stage required.
#
# Prereqs: container/llm-dev-env.sif built; data/method_a_qwen3b.jsonl(+heldout)
# present. Memory: full FT of 3B (bf16 model + AdamW states) ~55 GB, fits the
# 141 GB H200 with gradient checkpointing on.

module purge
module load apptainer/1.5.2

if [ -f .env ]; then set -a; source .env; set +a; fi
export HF_HOME="${HF_HOME:-/u/fash/.cache/huggingface}"
mkdir -p .log "$HF_HOME"

# Env-overridable so a throwaway smoke-train (code test only) can write to its
# own output dir instead of overwriting a real model. Defaults = the real run.
DATA="${DATA:-data/method_a_v2.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/method_a_v2_heldout.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-models/end-token-A-qwen3b-full}"
EPOCHS="${EPOCHS:-3}"
echo "train: data=$DATA | eval=$EVAL_DATA | out=$OUTPUT_DIR | epochs=$EPOCHS"

srun apptainer exec \
    --nv \
    --bind "$PWD":/workspace \
    --bind "${HF_HOME}":/root/.cache/huggingface \
    --env HF_HOME=/root/.cache/huggingface \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env WANDB_PROJECT="${WANDB_PROJECT:-end-token}" \
    container/llm-dev-env.sif \
    python scripts/fine_tuning/train_end_token.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        `# HF id (transformers), NOT the Ollama tag -- Ollama is only for harvest` \
        --full --gradient-checkpointing \
        --data "$DATA" \
        --eval-data "$EVAL_DATA" \
        --output-dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" --batch-size 2 --grad-accum 8 --max-length 2048
        `# LR defaults to 1e-5 for --full (see train_end_token.py)`
