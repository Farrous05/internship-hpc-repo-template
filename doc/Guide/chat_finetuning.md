# Chat Model Fine-tuning on HPC Clusters

This guide provides instructions to fine-tune language models specifically for chat applications on our HPC clusters (Raven and DAIS).

## Prerequisites
- Access to the HPC clusters (Raven and DAIS).
- Apptainer image for the LLM development environment. (see [dev-container.md](dev-container.md) for building instructions)
- (optional) Huggingface account and access token — only needed for gated models. [https://huggingface.co/docs/hub/en/security-tokens](https://huggingface.co/docs/hub/en/security-tokens)
- (optional) Wandb account and project for experiment tracking. Set `WANDB_API_KEY` in your `.env` file. [https://docs.wandb.ai/models/quickstart](https://docs.wandb.ai/models/quickstart)

## What is Chat Fine-tuning?

Chat fine-tuning trains a language model on conversation-style data so it learns to follow instructions and produce helpful responses. Unlike general language modeling, it focuses on the instruction → response pattern.

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that inserts small trainable weight matrices into the model instead of updating all parameters. This reduces memory usage by 10–100× and trains much faster — essential when working with models larger than GPT-2 on a single GPU.

The script uses LoRA by default via the `peft` library.

## Steps

1. **Prepare the Apptainer Image**: Ensure you have built the Apptainer image as described in [dev-container.md](dev-container.md).
2. Create a virtual environment inside the Apptainer container. Follow [hpc-usage.md](hpc-usage.md).
3. Copy `.env.example` to `.env` and fill in the required variables (`HF_TOKEN`, `WANDB_API_KEY` if using Wandb).
4. (Optional) Pre-download the dataset on the login node if the compute nodes lack internet access:
    ```bash
    python scripts/utils/download_dataset.py --path tatsu-lab/alpaca
    ```
5. Submit the fine-tuning job:
    ```bash
    # Raven
    sbatch scripts/fine_tuning/chat_finetuning_raven.slurm.sh

    # DAIS
    sbatch scripts/fine_tuning/chat_finetuning_dais.slurm.sh
    ```

## Running the Script Directly

The script accepts CLI arguments so you can customise it without editing code:

```bash
# Default: SmolLM2-135M-Instruct + Alpaca dataset
python scripts/fine_tuning/chat_finetuning.py

# Larger model with a cleaned Alpaca variant
python scripts/fine_tuning/chat_finetuning.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset yahma/alpaca-cleaned \
    --output-dir models/qwen-chat \
    --epochs 1

# All options
python scripts/fine_tuning/chat_finetuning.py --help
```

| Argument | Default | Description |
|---|---|---|
| `--model` | `HuggingFaceTB/SmolLM2-135M-Instruct` | HuggingFace model ID |
| `--dataset` | `tatsu-lab/alpaca` | HuggingFace dataset ID |
| `--output-dir` | `models/chat-finetuned` | Where to save the LoRA adapter |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `4` | Per-device batch size |
| `--max-length` | `512` | Max token length per example |

## Dataset Format

The default dataset (`tatsu-lab/alpaca`) uses this structure, which the script handles automatically:

```json
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris."
}
```

The `input` field is optional context appended to the instruction. The script formats examples using the model's built-in chat template via `apply_chat_template`, so the prompt format is always correct for the chosen model.

Any HuggingFace dataset with `instruction` and `output` columns will work.

## Output

After training, the `--output-dir` contains:
- `adapter_config.json` — LoRA configuration
- `adapter_model.safetensors` — trained adapter weights (much smaller than full model weights)
- `tokenizer_config.json` and related files

To load the adapter for inference:
```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("models/chat-finetuned")
tokenizer = AutoTokenizer.from_pretrained("models/chat-finetuned")
```

## Best Practices

- Use high-quality, diverse chat datasets
- Ensure sufficient training data (at least thousands of examples)
- Monitor training loss via Wandb to detect overfitting
- LoRA `r=8` is a good starting point; increase to `r=16` or `r=32` for harder tasks
- Use `--max-length 512` or higher for longer instruction-response pairs