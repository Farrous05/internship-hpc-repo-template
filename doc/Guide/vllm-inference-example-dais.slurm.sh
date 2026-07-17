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

export HF_HOME="/dais/fs/scratch/$USER/huggingface"

# create the HF cache directory if it does not exist
mkdir -p $HF_HOME

# run the inference server
srun apptainer exec \
  --nv \
  -B .:"$HOME",$HF_HOME \
  container/vllm.sif \
  vllm serve openai/gpt-oss-20b --port 8000 --host 0.0.0.0 > .log/inference_server.log 2>&1 &
SERVER_PID=$!

source .venv/bin/activate
python scripts/inference/vllm_inference_example.py

kill $SERVER_PID