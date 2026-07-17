#!/bin/bash -l

#SBATCH -o .log/job.out
#SBATCH -e .log/job.err
#SBATCH -D ./
#SBATCH -J llm_inference

#SBATCH --nodes=1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64GB

#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1

# Wall clock limit (max. is 24 hours):
#SBATCH --time=00:15:00

module purge
module load apptainer

export OLLAMA_MODELS="/dais/fs/scratch/$USER/ollama"

# create the models directory if it does not exist
mkdir -p $OLLAMA_MODELS

# run the inference server
srun apptainer run \
  --nv \
  -B .:"$HOME",$OLLAMA_MODELS \
  container/ollama.sif > .log/inference_server.log 2>&1 &
SERVER_PID=$!

source .venv/bin/activate
python scripts/inference/inference_example.py 

kill $SERVER_PID