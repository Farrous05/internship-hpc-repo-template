#!/bin/bash -l
#SBATCH --output {output_dir}/slurm-%x-%j.out
#SBATCH --error {output_dir}/slurm-%x-%j.out
#SBATCH --chdir ./
#SBATCH --job-name {job_name}/{job_id}
#
#SBATCH --nodes={n_nodes}
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --mem={memory}
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:{n_gpu}
#SBATCH --partition=gpu

# Wall clock limit (max is 24 hours):
#SBATCH --time={time}

module purge
module load apptainer

source .env

# create huggingface cache directory if it doesn't exist
mkdir -p ~/.cache/huggingface

echo "Runing training using the image: {image}"
echo "Runing training using the config: {config_file}"

srun apptainer exec \
	--nv \
    --contain \
    --cleanenv \
    --pwd /root/llm-strategic-tuning \
    --bind .:/root/llm-strategic-tuning \
    --bind ~/.cache/huggingface:/root/.cache/huggingface \
    --env HUGGING_FACE_HUB_TOKEN="$HUGGINGFACE_TOKEN" \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    --env WANDB_ENTITY="{entity_name}" \
    --env WANDB_PROJECT="{project_name}" \
    --env WANDB_RUN_GROUP="{group_name}" \
    --env WANDB_NAME="{job_name}/{job_id}" \
    --env NCCL_DEBUG="INFO" \
    --env NCCL_BLOCKING_WAIT="0" \
    --env HF_HOME="/root/.cache/huggingface" \
	{image} \
    python -m torch.distributed.run \
		--nnodes="$SLURM_NNODES" \
		--nproc-per-node=gpu \
		--rdzv-id="$SLURM_JOBID" \
		--rdzv-endpoint=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) \
		--rdzv-backend="c10d" \
    {script} --config {config_file}
