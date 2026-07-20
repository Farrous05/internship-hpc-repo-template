#!/bin/bash -l

#SBATCH -o .log/generate_healthy.out
#SBATCH -e .log/generate_healthy.err
#SBATCH -D /u/fash/internship-hpc-repo-template
#SBATCH -J gen-healthy

#SBATCH --nodes=1
#SBATCH --cpus-per-task 12
#SBATCH --mem 125GB
#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00

# Generate healthy (non-collapsed) two-voice conversations = the NEGATIVES.
#
# WHY GENERATE: collapsed data is free (100% of self-loop runs collapse) but
# healthy LLM-to-LLM data is the bottleneck -- only ~10% of deep pre-onset
# windows survive the judge. So we synthesize the half that is hard.
#
# MODEL CHOICE (both must be resident at once, 141GB H200):
#   generator = mixtral 8x7b q4  (~26GB) -- MoE, fast, diverse prose
#   judge     = gemma4:31b-it     (62GB) -- control-validated 95%/95%
#   total ~88GB, comfortable. (Llama-3.3-70B q8 is 74GB: 74+62=136GB is too
#   tight to co-locate with the judge, which is why it is not the generator here.)
#
# GENERATOR != JUDGE, deliberately: a model grading its own output favours it,
# which would make the healthy-verification worthless. The script enforces this.
#
# SEEDS MUST BE THE POOL THE POSITIVES ARE SEEDED FROM. A seed contributes only
# the SUBJECT: the generator writes the conversation from the seed's opening
# question. So the seed pool IS the list of subjects the negatives talk about --
# if it differs from the positives' pool, TOPIC predicts the class and the model
# learns the subject instead of "am I looping?".
#
# Previously hardcoded to sharegpt_real.json (400 ids), which overlaps the
# harvest's seeds_10k.json by only 37 -- a near-perfect topic shortcut. The
# held-out split is BY RUN, so it survives into the eval and inflates AUC
# without teaching the trigger. Default is now the harvest's own pool.

set -euo pipefail

GEN_MODEL="${GEN_MODEL:-mixtral:8x7b-instruct-v0.1-q4_K_M}"
JUDGE_MODEL="${JUDGE_MODEL:-gemma4:31b-it-bf16}"
N="${N:-400}"                 # generate this many; ~judge-pass rate survives
SEEDS="${SEEDS:-/u/fash/babel-ai/data/seeds_10k.json}"
OUT_FILE="${OUT_FILE:-data/healthy_generated.json}"
WORKERS="${WORKERS:-4}"       # measured: N=400 in 44 min at 4 (job 364404)
TURNS="${TURNS:-6}"
PORT=11434
# This job now LIVES in the template (owns generate_healthy.py + its output), but
# the Ollama container + models still live in babel-ai/scratch, by absolute path.
# cwd (SBATCH -D) is the template, so `scripts/...` + `data/...` resolve here.
BABEL=/u/fash/babel-ai

module purge
module load apptainer/1.5.2

export OLLAMA_MODELS="/dais/fs/scratch/$USER/ollama"
export OLLAMA_CONTEXT_LENGTH=16384
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2   # generator + judge resident together
mkdir -p .log

# bind babel-ai as $HOME so Ollama's runtime layout is identical to before
srun apptainer run --nv -B "$BABEL":"$HOME",$OLLAMA_MODELS "$BABEL/container/ollama.sif" \
    > .log/ollama_gen_healthy.log 2>&1 &
SERVER_PID=$!
trap 'echo "stopping Ollama"; kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 90); do
    curl -sf "http://localhost:$PORT/api/tags" >/dev/null 2>&1 && { echo "Ollama up"; break; }
    kill -0 $SERVER_PID 2>/dev/null || { echo "Ollama died"; exit 1; }
    sleep 5
done
for M in "$GEN_MODEL" "$JUDGE_MODEL"; do
    apptainer exec -B $OLLAMA_MODELS "$BABEL/container/ollama.sif" ollama pull "$M" || \
        echo "pull skipped for $M - assuming cached"
done

source .venv/bin/activate
# the scripts speak OpenAI-protocol; Ollama serves an OpenAI-compatible API, so
# these two env vars are the whole "port to local models" change.
export OPENAI_BASE_URL="http://localhost:$PORT/v1"
export OPENAI_API_KEY="ollama"

echo "=== generating $N healthy conversations ==="
echo "    generator: $GEN_MODEL"
echo "    judge:     $JUDGE_MODEL"
echo "    seeds:     $SEEDS  (must match the harvest's seed pool)"
echo "    out:       $OUT_FILE"
python scripts/generate_healthy.py \
    --n "$N" \
    --model "$GEN_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --seeds "$SEEDS" \
    --turns "$TURNS" --workers "$WORKERS" \
    --out "$OUT_FILE"

echo "done -> $REPO/$OUT_FILE"
