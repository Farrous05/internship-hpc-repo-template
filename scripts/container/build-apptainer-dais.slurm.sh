#!/bin/bash -l

#SBATCH -o .log/job.out
#SBATCH -e .log/job.err
#SBATCH -D ./
#SBATCH -J build_llm_dev_env

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1

# Wall clock limit (max. is 24 hours):
#SBATCH --time=03:00:00

module purge
module load apptainer

# Create a temporary directory for Apptainer
TMP=$(mktemp -p /u/$(whoami)/.cache/ -d)
export APPTAINER_TMPDIR=$TMP

rm -rf container/llm-dev-env.sif
apptainer build container/llm-dev-env.sif container/llm-dev-env.def

rm -rf $TMP
