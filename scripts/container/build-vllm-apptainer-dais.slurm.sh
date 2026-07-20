#!/bin/bash -l

#SBATCH -o .log/build_vllm.out
#SBATCH -e .log/build_vllm.err
#SBATCH -D ./
#SBATCH -J build_vllm

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

#SBATCH --partition="gpu1"
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks-per-node=1

# Wall clock limit (max. is 24 hours):
#SBATCH --time=03:00:00

# Build container/vllm.sif for serving HF models (incl. our <end>-token model)
# via an OpenAI-compatible API. DAIS is NVIDIA, so we use the official prebuilt
# vllm/vllm-openai image -- no %post/compilation, just pull + convert (mirrors
# the guide in doc/Guide/llm-inference.md; the .sh it names was never committed).
# The image is prebuilt, so this needs no GPU to build, but we mirror the known-
# good build-apptainer-dais.slurm.sh allocation to stay on the supported path.

module purge
module load apptainer

# Apptainer needs a roomy tmpdir: the image is multi-GB and the default /tmp
# often overflows mid-convert.
TMP=$(mktemp -p /u/$(whoami)/.cache/ -d)
export APPTAINER_TMPDIR=$TMP

rm -rf container/vllm.sif
apptainer build container/vllm.sif docker://vllm/vllm-openai:latest

rm -rf $TMP
echo "done -> container/vllm.sif"
ls -la container/vllm.sif