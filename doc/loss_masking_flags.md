# Loss masking in `train_end_token.py` — notes & caveats

Where: [`make_tokenize_fn`](../scripts/fine_tuning/train_end_token.py) (the `tok`
closure). The script does **completion-only** training — loss is computed only
over the target assistant turn; the conversation history is masked out with
`-100`.

## How the mask is built

```python
prompt = apply_chat_template(messages,            add_generation_prompt=True)   # history + assistant header
full   = apply_chat_template(messages + [target], add_generation_prompt=False)  # history + full target turn
full   = full[:max_length]
labels = list(full)
n_prompt = min(len(prompt), len(full))
for i in range(n_prompt):
    labels[i] = -100        # -100 = "ignore this position in the loss"
```

`prompt` is (meant to be) an exact token prefix of `full`; `len(prompt)` is
therefore the index where the assistant reply begins. Everything before it is
blanked to `-100`, so 100% of the learning pressure lands on the reply — and in
particular on whether the reply's **first token is `<end>`**.

## Flags / caveats

### 1. The mask boundary is positional, not verified
The code trusts that `prompt` is a token-exact prefix of `full` and simply blanks
the first `len(prompt)` tokens. It never checks that the tokens actually match. For
ChatML models (Qwen2.5, SmolLM) this holds. But some chat templates emit slightly
different tokens when `add_generation_prompt=True` vs. when a real assistant turn
follows (an extra newline, a system-prompt tweak). If that happened, the boundary
would be off by a token or two — either grading a stray history token or ignoring
the first reply token (which is the `<end>` decision). Works here; unchecked in
general.

### 2. The collator choice is load-bearing
Batching pads sequences to equal length via a "collator." The script uses
`DataCollatorForSeq2Seq`, which pads `labels` with `-100` and **keeps** the mask.
The more common `DataCollatorForLanguageModeling` would discard the labels and
re-set `labels = input_ids`, grading the model on everything (history included) and
silently destroying the masking. The choice of collator is what preserves the
completion-only behavior.

### 3. Train/eval truncation asymmetry
Sequences over `max_length` are cut from **opposite ends**:
- **Train**: `full[:max_length]` cuts from the **right** — a long history can chop
  off the target reply, leaving all `-100`; those rows are then dropped by the
  `has_signal` filter.
- **Eval**: `ids[-max_length:]` cuts from the **left** — deliberately keeping the
  end of the conversation, where the `<end>` decision happens.

Not a bug (the `TURN_CHAR_CAP=500` cap in `build_offline_dataset.py` keeps most
examples short enough that this rarely triggers), but the two sides handle overflow
differently and it's worth knowing.
