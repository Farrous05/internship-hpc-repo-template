"""Method A — offline SFT to teach the <end> token (project Phase 2c).

Two modes:
  * default = LoRA (for the Qwen2.5-72B target: cheap, freezes the base).
  * --full  = FULL fine-tune (for the Qwen2.5-3B PoC: every weight trains, so the
    new <end> embedding rows train naturally without LoRA's modules_to_save; use
    a much lower LR and warmup, and gradient checkpointing to fit the H200).

Modeled on the company chat-finetuning guide (peft, apply_chat_template, Trainer),
with three task-specific additions the guide's script does not have:

  1. **Add the <end> token** to the tokenizer and resize the embeddings. Because
     LoRA freezes the base weights, the *new* embedding rows would never train --
     so we add ``embed_tokens`` and ``lm_head`` to LoRA ``modules_to_save`` to
     make exactly those trainable. (Without this the model can never learn to
     emit <end>.)
  2. **Read our dataset**: JSONL rows of {"messages": [...history...],
     "target": {"role":"assistant","content": "..."}}, not Alpaca
     instruction/output.
  3. **Mask the loss to the target turn** (completion-only): the history tokens
     get label -100, so the model learns *when* to produce the next turn (and
     whether it starts with <end>), not to memorize conversation histories.

Runs on the llm-dev-env.sif image. Defaults to a tiny model so the pipeline can
be smoke-tested on CPU before committing gemma to the H200.

Usage:
  python scripts/fine_tuning/train_end_token.py \
      --model google/gemma-4-31B-it \
      --data data/offline_dataset.jsonl \
      --eval-data data/offline_dataset_heldout.jsonl \
      --output-dir models/end-token-A
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

END_TOKEN = "<end>"
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"  # tiny -> CPU smoke test


def parse_args():
    p = argparse.ArgumentParser(description="Method A: offline SFT for <end>")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", required=True, help="train JSONL (messages+target)")
    p.add_argument("--eval-data", default=None, help="held-out JSONL")
    p.add_argument("--output-dir", default="models/end-token-A")
    p.add_argument("--full", action="store_true",
                   help="full fine-tune (no LoRA); for the 3B PoC")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="trade compute for memory (recommended with --full)")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--lr", type=float, default=None,
                   help="default 1e-5 for --full, else 2e-4 (LoRA)")
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args()


def load_model_and_tokenizer(model_id, full, lora_r, lora_alpha):
    """Load model + tokenizer, add <end>, resize. In LoRA mode, wrap in PEFT with
    the new embedding rows made trainable; in --full mode, leave every weight
    trainable (the resized rows train on their own)."""
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1) add <end> (no-op if the tokenizer already has it)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [END_TOKEN]}
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    )
    if added:
        # mean-init the new row(s) so <end> starts near the embedding manifold
        model.resize_token_embeddings(len(tokenizer), mean_resizing=True)

    if full:
        # 2a) full fine-tune: nothing frozen; embed_tokens (incl. <end>) and
        # lm_head train like any other weight. No PEFT wrapper.
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"full fine-tune: {n/1e9:.2f}B trainable params")
        return model, tokenizer

    # 2b) LoRA. Let PEFT pick the modules ("all-linear" = every linear layer,
    # attention + MLP) so we don't hardcode names. embed_tokens + lm_head go in
    # modules_to_save so the resized rows (incl. <end>) actually train under LoRA.
    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


def make_tokenize_fn(tokenizer, max_length):
    """Format one row and mask the loss to the target turn only."""
    def tok(example):
        messages = example["messages"]
        target = example["target"]
        # return_dict=False is REQUIRED: since transformers 5.x
        # apply_chat_template(tokenize=True) returns a BatchEncoding, not the
        # List[int] this code slices and masks. Without it, list(full) yields
        # the KEY NAMES ['input_ids', 'attention_mask'] and an unserialisable
        # Encoding reaches pyarrow ("did not recognize Python value type").
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=False,
        )
        # full = history + the target assistant turn
        full = tokenizer.apply_chat_template(
            messages + [target], tokenize=True, add_generation_prompt=False,
            return_dict=False,
        )
        full = full[:max_length]
        labels = list(full)
        # everything before the target turn is context -> ignore in the loss
        n_prompt = min(len(prompt), len(full))
        for i in range(n_prompt):
            labels[i] = -100
        return {"input_ids": full, "attention_mask": [1] * len(full),
                "labels": labels}
    return tok


def main():
    load_dotenv()
    args = parse_args()

    lr = args.lr if args.lr is not None else (1e-5 if args.full else 2e-4)
    model, tokenizer = load_model_and_tokenizer(
        args.model, args.full, args.lora_r, args.lora_alpha
    )

    tok = make_tokenize_fn(tokenizer, args.max_length)
    # drop examples whose target got truncated to nothing (all labels -100):
    # they carry no gradient and a full batch of them yields NaN loss (I2).
    def has_signal(ex):
        return any(x != -100 for x in ex["labels"])

    train = load_dataset("json", data_files=args.data, split="train")
    train = train.map(tok, remove_columns=train.column_names).filter(has_signal)
    eval_ds = None
    if args.eval_data and os.path.exists(args.eval_data):
        eval_ds = load_dataset("json", data_files=args.eval_data, split="train")
        eval_ds = eval_ds.map(tok, remove_columns=eval_ds.column_names).filter(has_signal)

    report_to = "wandb" if os.getenv("WANDB_API_KEY") else "none"
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=10,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        run_name="end-token-A" + ("-full" if args.full else "-lora"),
        report_to=report_to,
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train, eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    kind = "full model" if args.full else "LoRA adapter"
    print(f"{kind} + tokenizer (with {END_TOKEN}) saved to {args.output_dir}")


if __name__ == "__main__":
    main()