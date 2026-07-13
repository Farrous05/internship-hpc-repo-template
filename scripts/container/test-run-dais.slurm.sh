#!/bin/bash -l
#
#SBATCH -o .log/job.out
#SBATCH -e .log/job.err
#SBATCH -D ./
#SBATCH -J llm-dev-env-test

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1


# Wall clock limit (max. is 24 hours):
#SBATCH --time=01:00:00

module purge
module load apptainer

echo "[$(date)] Starting inference server..."
# run the inference server
srun apptainer exec \
  --nv \
  --bind $PWD:/workspace \
  container/llm-dev-env.sif \
  python scripts/container/test_run.py
