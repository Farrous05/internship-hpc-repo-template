# LLM Inference on HPC Clusters
This guide provides instructions to run LLM inference on our DAIS/Raven clusters, using either [Ollama](#inference-using-ollama) or [vLLM](#inference-using-vllm).

## Inference using Ollama

### Prerequisites
- Access to the HPC clusters (Raven and DAIS).
- Have `uv` installed in your local machine. You can follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/)

### Steps
1. **Prepare Python Virtual Environment**: Run this command to create a Python virtual environment with the required packages.
    ```bash
    uv sync
    ```
    You should have `.venv` folder created in the current directory as well as python packages installed in it.
You can then activate the virtual environment by running:
    ```bash
    source .venv/bin/activate
    ```

2. **Prepare Ollama Apptainer Image**: Build the Apptainer image for Ollama by using the provided script `scripts/container/build-ollama-apptainer-dais.sh`. Submit this script as a job on the cluster to create the image.
    ```bash
    # if on DAIS
    sbatch scripts/container/build-ollama-apptainer-dais.sh

    # if on Raven
    sbatch scripts/container/build-ollama-apptainer-raven.sh
    ```
    By the end of the job, you should have `container/ollama.sif` file created.

2. **Check on the inference script**: You can find the inference script at `scripts/inference/inference_example.sh`. Here we will be using `openai` library to interact with the Ollama model. Make sure to adjust the parameters such as model name, input prompt, and output file according to your needs.

3. **Create a Slurm Job Script**: Create a Slurm job script to run the inference process using the Apptainer image. You can use the provided `scripts/inference/llm_inference_*.slurm.sh` as a template. Make sure to adjust the parameters such as job name, output files, and resource allocation according to your needs.
4. **Submit the Job**: Finally, submit the Slurm job script to the cluster using the `sbatch` command.
    ```bash
    # if on DAIS
    sbatch scripts/inference/llm_inference_dais.slurm.sh
    # if on Raven
    sbatch scripts/inference/llm_inference_raven.slurm.sh
    ```
5. **Observe the Output**: Once the job is completed, check the output files specified in your Slurm job script to see the inference results. By default, the output will be saved in `.log/jobs.out`.

## Inference using vLLM
[vLLM](https://github.com/vllm-project/vllm) is a high-throughput, OpenAI-compatible LLM serving engine. Unlike Ollama, it serves HuggingFace models directly (no separate "pull" step) and doesn't require a `docker://ollama/ollama` style tag per model — you just pass the model name to `vllm serve` at startup.

This guide covers three clusters: **Raven** and **DAIS** (NVIDIA GPUs, using the `vllm/vllm-openai` image), and **Viper** (AMD MI300A APUs, using the ROCm-based `rocm/vllm` image instead).

### Prerequisites
- Access to the HPC clusters (Raven, DAIS, and/or Viper).
- Have `uv` installed in your local machine. You can follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/)
- A HuggingFace token (`HF_TOKEN`) if you plan to serve a gated model.

### Steps
1. **Prepare Python Virtual Environment**: Same as the Ollama section above — run `uv sync` and activate the resulting `.venv`.

2. **Prepare vLLM Apptainer Image**: Build the Apptainer image for vLLM by using the provided script `scripts/container/build-vllm-apptainer-dais.sh` (or the Raven equivalent). Submit this script as a job on the cluster to create the image.
    ```bash
    # if on DAIS
    sbatch scripts/container/build-vllm-apptainer-dais.sh

    # if on Raven
    sbatch scripts/container/build-vllm-apptainer-raven.sh

    # if on Viper
    sbatch scripts/container/build-vllm-apptainer-viper.sh
    ```
    By the end of the job, you should have a `container/vllm.sif` file created. On Viper, this pulls the ROCm-based `rocm/vllm` image instead of `vllm/vllm-openai`.

3. **Check on the inference script**: You can find the inference script at `scripts/inference/vllm_inference_example.py`. Here we will be using the `openai` library to interact with the vLLM server. Make sure to adjust the parameters such as model name, input prompt, and output handling according to your needs.

4. **Submit the Slurm Job**: Use the provided `scripts/inference/vllm-inference-example-*.slurm.sh` scripts, which start the vLLM server (`vllm serve openai/gpt-oss-20b --port 8000`) inside the Apptainer image and then run the inference script against it. Make sure to adjust parameters such as job name, output files, resource allocation, and the served model according to your needs.
    ```bash
    # if on DAIS
    sbatch scripts/inference/vllm-inference-example-dais.slurm.sh
    # if on Raven
    sbatch scripts/inference/vllm-inference-example-raven.slurm.sh

    # if on Viper
    sbatch scripts/inference/vllm-inference-example-viper.slurm.sh
    ```

5. **Observe the Output**: Once the job is completed, check the output files specified in your Slurm job script to see the inference results (`.log/job.out` on DAIS, `job.out.%j` in the working directory on Raven/Viper), with the vLLM server logs in `.log/inference_server.log`.