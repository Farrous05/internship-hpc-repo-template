#!/bin/bash -l

#SBATCH -o .log/eval_end_token.out
#SBATCH -e .log/eval_end_token.err
#SBATCH -D /u/fash/internship-hpc-repo-template
#SBATCH -J eval-end-token

#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=1
#SBATCH --mem=125GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00

# Eval the <end>-token model: TRIGGER (AUC on P(<end>)), RETENTION (perplexity
# base vs tuned), NO-SPAM (generate on healthy, count <end>). Env-overridable so
# the same script serves the throwaway smoke model and the real Method A run.
#   MODEL      = the fine-tuned model dir
#   EVAL_DATA  = held-out jsonl
#   BASE_MODEL = untuned model for the retention comparison

module purge
module load apptainer/1.5.2

if [ -f .env ]; then set -a; source .env; set +a; fi
export HF_HOME="${HF_HOME:-/u/fash/.cache/huggingface}"
mkdir -p .log "$HF_HOME"

MODEL="${MODEL:-models/end-token-A-qwen3b-full}"
EVAL_DATA="${EVAL_DATA:-data/method_a_v2_heldout.jsonl}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
OUT="${OUT:-data/eval_report.json}"
echo "eval: model=$MODEL | eval=$EVAL_DATA | base=$BASE_MODEL"

srun apptainer exec \
    --nv \
    --bind "$PWD":/workspace \
    --bind "${HF_HOME}":/root/.cache/huggingface \
    --env HF_HOME=/root/.cache/huggingface \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    container/llm-dev-env.sif \
    python scripts/eval_end_token.py \
        --model "$MODEL" \
        --eval-data "$EVAL_DATA" \
        --base-model "$BASE_MODEL" \
        --out "$OUT"
