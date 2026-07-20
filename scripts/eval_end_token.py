"""Evaluate the <end> token model (project Phase 2d) -- the PoC's verdict.

THREE TESTS
  1. TRIGGER (the headline). The model's entire decision is "is my next token
     <end>?". So we do ONE forward pass per held-out example and read P(<end>) at
     the first generated position -- no sampling, no generation, deterministic.
     Positives (loops) should score HIGH, negatives (healthy) LOW. We report AUC,
     which needs no hand-picked threshold:
        AUC ~0.5 -> learned nothing.   AUC ~1.0 -> learned the behaviour.
  2. RETENTION. Perplexity on normal text, base vs trained. A FULL fine-tune moves
     every weight, so this is where we catch the model getting worse at ordinary
     conversation -- one of the two primary metrics.
  3. NO SPAM. Actually generate on normal prompts and count <end>. Should be ~0.
     (A model that fires <end> everywhere would still score well on test 1 if we
     only looked at positives -- this is the check that catches it.)

Usage:
  python scripts/eval_end_token.py --model models/end-token-A-qwen3b-full \
      --eval-data data/method_a_qwen3b_heldout.jsonl \
      [--base-model Qwen/Qwen2.5-3B-Instruct] [--max-length 2048]
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

END_TOKEN = "<end>"


def p_end(model, tok, end_id, messages, max_length):
    """P(<end>) as the FIRST token of the assistant's reply = the decision point."""
    ids = tok.apply_chat_template(messages, tokenize=True,
                                  add_generation_prompt=True, return_dict=False)
    ids = ids[-max_length:]                     # keep the END (the decision point)
    x = torch.tensor([ids], device=model.device)
    with torch.no_grad():
        logits = model(x).logits[0, -1]          # next-token distribution
    return torch.softmax(logits.float(), dim=-1)[end_id].item()


def auc(pos, neg):
    """Probability a random positive scores above a random negative (Mann-Whitney).
    Threshold-free, so it can't be gamed by picking a lucky cutoff."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def perplexity(model, tok, texts, max_length):
    """Mean token-level perplexity -- lower is better. Used base-vs-trained."""
    tot_nll, tot_tok = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_length).input_ids.to(model.device)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        tot_nll += out.loss.item() * n
        tot_tok += n
    return float(torch.exp(torch.tensor(tot_nll / max(tot_tok, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the fine-tuned model dir")
    ap.add_argument("--eval-data", required=True, help="held-out JSONL")
    ap.add_argument("--base-model", default=None,
                    help="untuned model, for the retention comparison")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--n-generate", type=int, default=20)
    ap.add_argument("--out", default="data/eval_report.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval_data)]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    end_id = tok.convert_tokens_to_ids(END_TOKEN)
    if end_id is None or end_id == tok.unk_token_id:
        raise SystemExit(f"{END_TOKEN} not in tokenizer -- wrong model dir?")

    # ---- TEST 1: trigger -----------------------------------------------------
    pos, neg, neg_by_src = [], [], {}
    for r in rows:
        p = p_end(model, tok, end_id, r["messages"], args.max_length)
        if r["label_type"] == "positive":
            pos.append(p)
        else:
            neg.append(p)
            neg_by_src.setdefault(r.get("source", "?"), []).append(p)

    def mean(v):
        return sum(v) / len(v) if v else float("nan")

    a = auc(pos, neg)
    # fire rate at a plain 0.5 cutoff: what you'd actually see at inference
    fired_pos = sum(p > 0.5 for p in pos) / max(len(pos), 1)
    fired_neg = sum(p > 0.5 for p in neg) / max(len(neg), 1)

    print("=== TEST 1: TRIGGER (does P(<end>) separate loops from healthy?) ===")
    print(f"  positives n={len(pos):3d}  mean P(<end>) = {mean(pos):.3f}")
    print(f"  negatives n={len(neg):3d}  mean P(<end>) = {mean(neg):.3f}")
    for s, v in neg_by_src.items():
        print(f"     └ {s:20s} n={len(v):3d}  mean P = {mean(v):.3f}")
    print(f"  AUC = {a:.3f}   (0.5 = learned nothing, 1.0 = perfect separation)")
    print(f"  at threshold 0.5: fires on {100*fired_pos:.0f}% of loops, "
          f"{100*fired_neg:.0f}% of healthy  (want high / low)")

    # ---- TEST 3: no spam -----------------------------------------------------
    # (run before we load the base model, so we only hold one model at a time)
    healthy = [r for r in rows if r["label_type"] == "negative"][:args.n_generate]
    spam, gens = 0, 0
    for r in healthy:
        ids = tok.apply_chat_template(
            r["messages"], tokenize=True, add_generation_prompt=True,
            return_dict=False)[-args.max_length:]
        with torch.no_grad():
            out = model.generate(torch.tensor([ids], device=model.device),
                                 max_new_tokens=80, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        txt = tok.decode(out[0][len(ids):], skip_special_tokens=False)
        gens += 1
        spam += END_TOKEN in txt
    print("\n=== TEST 3: NO SPAM (generate on healthy prompts) ===")
    print(f"  {spam}/{gens} generations contained {END_TOKEN}   (want ~0)")

    # ---- TEST 2: retention ---------------------------------------------------
    ppl_texts = [m["content"] for r in rows if r.get("source") == "sharegpt"
                 for m in r["messages"]][:60]
    report = {"auc": a, "mean_p_pos": mean(pos), "mean_p_neg": mean(neg),
              "fire_rate_pos@0.5": fired_pos, "fire_rate_neg@0.5": fired_neg,
              "spam": f"{spam}/{gens}"}
    if ppl_texts:
        ppl_tuned = perplexity(model, tok, ppl_texts, args.max_length)
        report["ppl_tuned"] = ppl_tuned
        print("\n=== TEST 2: RETENTION (perplexity on normal text, lower=better) ===")
        print(f"  tuned model: {ppl_tuned:.2f}")
        if args.base_model:
            del model
            torch.cuda.empty_cache()
            base = AutoModelForCausalLM.from_pretrained(
                args.base_model, torch_dtype=torch.bfloat16,
                device_map="auto").eval()
            btok = AutoTokenizer.from_pretrained(args.base_model)
            ppl_base = perplexity(base, btok, ppl_texts, args.max_length)
            report["ppl_base"] = ppl_base
            delta = 100 * (ppl_tuned - ppl_base) / ppl_base
            print(f"  base  model: {ppl_base:.2f}")
            print(f"  change: {delta:+.1f}%   (large increase = capability damage)")

    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    print("\n=== VERDICT ===")
    # AUC alone is NOT enough: P(<end>) can be ~1e-9 on positives and ~1e-11 on
    # negatives -> AUC=1.0 by an infinitesimal margin while the model NEVER fires
    # (fired_pos=0). A real PASS needs the model to actually EMIT <end> on loops,
    # so gate on the absolute fire rate too, not just the ranking.
    if fired_pos < 0.10:
        print(f"  FAIL: the model barely fires <end> at all "
              f"(fires on {100*fired_pos:.0f}% of loops, mean P={mean(pos):.2e}). "
              f"AUC={a:.3f} here is a near-zero-probability artifact, not learning. "
              f"Undertrained or too few positives; <end> was ~0.4% of the signal "
              f"-- consider more data, more epochs, or trigger-region loss masking.")
    elif a > 0.9 and fired_pos > 0.5 and fired_neg < 0.2:
        print(f"  PASS: fires on {100*fired_pos:.0f}% of loops, "
              f"{100*fired_neg:.0f}% of healthy; AUC={a:.3f}.")
    elif a > 0.7:
        print("  PARTIAL: some signal, but the trigger is mushy. Check the "
              "negatives (too few? contaminated?) and the loss balance.")
    else:
        print("  FAIL: no real separation. <end> was ~0.4% of the training signal "
              "-- consider trigger-region loss masking, and more negatives.")


if __name__ == "__main__":
    main()
