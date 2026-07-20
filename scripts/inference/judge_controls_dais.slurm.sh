#!/bin/bash -l

#SBATCH -o .log/judge_controls.out
#SBATCH -e .log/judge_controls.err
#SBATCH -D /u/fash/internship-hpc-repo-template
#SBATCH -J judge-controls

#SBATCH --nodes=1
#SBATCH --cpus-per-task 12
#SBATCH --mem 125GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

# Validate a LOCAL model as the loop/healthy judge, against BOTH pre-registered
# controls. We moved off the OpenAI judge because that key ran out of quota, and
# because judging ~20k windows per rebuild (the 10k-run plan) must be free.
#
# The judge scripts talk OpenAI-protocol; Ollama serves an OpenAI-compatible API,
# so pointing OPENAI_BASE_URL at localhost is enough -- no code change.
#
# Pass/fail bars (set BEFORE running, same as the gpt-4o run):
#   CONTROL A (known collapse-onset windows) -> must be >=90% "looping"
#   CONTROL B (known-healthy ShareGPT chat)  -> must be >=85% "progressing"
# gpt-4o scored 97% / 90%. gpt-4o-mini scored 70% on A and was REJECTED.
#
# Usage:  sbatch --export=ALL,JUDGE_MODEL=gemma4:31b-it-bf16 scripts/inference/judge_controls_dais.slurm.sh
#         sbatch --export=ALL,JUDGE_MODEL=llama3.3:70b-instruct-q8_0 ...

set -euo pipefail

MODEL="${JUDGE_MODEL:-gemma4:31b-it-bf16}"
PORT=11434
# This job now LIVES in the template (it owns the judge python + its output), but
# the Ollama container + models still live in babel-ai/scratch, referenced by
# absolute path. cwd (SBATCH -D) is the template, so `scripts/...` + `data/...`
# resolve here directly -- no `cd` needed.
BABEL=/u/fash/babel-ai

module purge
module load apptainer/1.5.2

export OLLAMA_MODELS="/dais/fs/scratch/$USER/ollama"
export OLLAMA_CONTEXT_LENGTH=16384
export OLLAMA_NUM_PARALLEL=4
mkdir -p .log

# 1. serve the judge model locally (bind babel-ai as $HOME so Ollama's runtime
#    layout is identical to when this job lived there)
srun apptainer run --nv -B "$BABEL":"$HOME",$OLLAMA_MODELS "$BABEL/container/ollama.sif" \
    > .log/ollama_judge.log 2>&1 &
SERVER_PID=$!
trap 'echo "stopping Ollama"; kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 90); do
    if curl -sf "http://localhost:$PORT/api/tags" >/dev/null 2>&1; then
        echo "Ollama up after ${i} checks"; break
    fi
    kill -0 $SERVER_PID 2>/dev/null || { echo "Ollama died - see .log/ollama_judge.log"; exit 1; }
    sleep 5
done
apptainer exec -B $OLLAMA_MODELS "$BABEL/container/ollama.sif" ollama pull "$MODEL" || \
    echo "pull skipped - assuming cached on scratch"

# 2. run BOTH controls against it. OPENAI_BASE_URL routes the OpenAI client at
#    the local server; the api key is a placeholder Ollama ignores.
source .venv/bin/activate
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="ollama"

echo "=== validating judge model: $MODEL ==="
python scripts/judge_windows.py \
    "results,/u/fash/babel-ai/results/qwen3b_harvest" \
    --n 40 --model "$MODEL" \
    --sharegpt /u/fash/babel-ai/data/sharegpt_real.json \
    --out "data/judged_local_${MODEL//[:\/]/_}.json"

echo "judge validation done for $MODEL"
