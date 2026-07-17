"""LLM judge: is a conversation window PROGRESSING or LOOPING?

Why: word-overlap / cosine thresholds conflate "same topic" (healthy) with "same
content" (collapse) -- two healthy turns about one Python function share most of
their vocabulary. So a metric cannot reliably find healthy LLM-to-LLM windows.
This asks a model the semantic question instead, and gives a detector-INDEPENDENT
label (which also addresses the circularity of labeling and evaluating with the
same detector).

Robustness: every run includes a CONTROL group -- windows at the known collapse
onset, which the judge MUST call "looping". If control accuracy is low, the judge
is not trustworthy and its other verdicts should be ignored.

Window types judged:
  positive_control : window ending at the collapse onset      -> expect LOOPING
  deep_preonset    : deep window, far before onset (late runs)-> healthy candidate
  first5           : iterations 0-4 (incl. the fetcher seed)  -> healthy candidate

Usage:
  OPENAI_API_KEY=... python scripts/judge_windows.py RESULTS_DIRS [--n 60]
      [--model gpt-4o-mini] [--out judged.json]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import random
import sys

def _load(name, path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(__file__), path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_b = _load("b", "build_offline_dataset.py")
_j = _load("j", "judge_lib.py")   # the single validated prompt lives there

RNG = random.Random(0)
W = 5                 # turns per window (matches the dataset's W_CTX)
TURN_CAP = 500        # match the dataset's history cap
DEEP_MIN = 6          # a "deep" window must end at/after this iteration
PREONSET_MARGIN = 6   # ...and this far before onset
LATE_ONSET = 16       # deep_preonset only from runs collapsing this late

def sharegpt_control(path, n):
    """KNOWN-HEALTHY control: real human<->assistant chat, definitionally not
    collapsed. The judge should call these 'progressing'. This tests the OTHER
    error direction -- positive_control only proves it catches loops, not that it
    recognises healthy conversation. If this scores low, the judge is over-strict
    and the low healthy-yield elsewhere is an artifact."""
    data = json.load(open(path))
    out = []
    for conv in data:
        items = [m for m in (conv.get("items") or [])
                 if m.get("from") in ("human", "gpt")]
        if len(items) < W:
            continue
        turns = [m.get("value") or "" for m in items[:W]]
        if any(not t.strip() for t in turns):
            continue
        out.append(dict(run_id="sharegpt:" + str(conv.get("id", "?")),
                        iters=[0, W - 1], turns=turns))
    RNG.shuffle(out)
    return out[:n]


def candidates(meta_paths, n_per_type):
    out = {"positive_control": [], "deep_preonset": [], "first5": []}
    for m in meta_paths:
        if "BUGGYSEED" in m:
            continue
        run = _b.load_run(m)
        if not run or run["onset"] is None:
            continue
        o, c = run["onset"], run["contents"]
        if o >= W and o < len(c):
            out["positive_control"].append(
                dict(run_id=run["run_id"], iters=[o - W + 1, o], turns=c[o - W + 1:o + 1]))
        if len(c) >= W:
            out["first5"].append(
                dict(run_id=run["run_id"], iters=[0, W - 1], turns=c[0:W]))
        if o >= LATE_ONSET:
            hi = o - PREONSET_MARGIN
            lo = max(W - 1, DEEP_MIN)
            if hi > lo:
                t = RNG.randint(lo, hi - 1)
                out["deep_preonset"].append(
                    dict(run_id=run["run_id"], iters=[t - W + 1, t], turns=c[t - W + 1:t + 1]))
    for k in out:
        RNG.shuffle(out[k])
        out[k] = out[k][:n_per_type]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dirs", help="comma-separated results dirs")
    ap.add_argument("--n", type=int, default=60, help="windows per type")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--sharegpt", default=None,
                    help="ShareGPT json -> adds a KNOWN-HEALTHY control group")
    ap.add_argument("--out", default="data/judged_windows.json")
    args = ap.parse_args()

    metas = []
    for rd in args.results_dirs.split(","):
        metas += glob.glob(os.path.join(rd.strip(), "**", "*_meta.json"),
                           recursive=True)
    metas = sorted(set(metas))
    cands = candidates(metas, args.n)
    if args.sharegpt:
        cands["sharegpt_control"] = sharegpt_control(args.sharegpt, args.n)
    print("candidate windows: " +
          " ".join(f"{k}={len(v)}" for k, v in cands.items()), flush=True)

    from openai import OpenAI
    client = OpenAI()
    results = {}
    for kind, items in cands.items():
        verdicts = []
        for i, it in enumerate(items):
            v = _j.judge(client, args.model, it["turns"])
            it["verdict"] = v.get("verdict")
            it["confidence"] = v.get("confidence")
            it["reason"] = v.get("reason")
            verdicts.append(it["verdict"])
            print(f"  {kind} {i+1}/{len(items)}: {it['verdict']}", flush=True)
        n = len(verdicts) or 1
        loop = verdicts.count("looping")
        prog = verdicts.count("progressing")
        results[kind] = dict(n=len(verdicts), looping=loop, progressing=prog,
                             pct_progressing=round(100 * prog / n))
    # strip turns for a compact dump
    for k, items in cands.items():
        for it in items:
            it["turns"] = [t[:200] for t in it["turns"]]
    json.dump({"summary": results, "windows": cands}, open(args.out, "w"), indent=1)

    print("\n=== JUDGE SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:16s} n={v['n']:3d}  looping={v['looping']:3d}  "
              f"progressing={v['progressing']:3d}  ({v['pct_progressing']}% progressing)")
    ctrl = results.get("positive_control", {})
    if ctrl.get("n"):
        acc = 100 * ctrl["looping"] // ctrl["n"]
        print(f"\nCONTROL A (known loops): judge called {acc}% 'looping'.")
        print("  >=90% = catches loops; <80% = do NOT trust its verdicts.")
    sg = results.get("sharegpt_control", {})
    if sg.get("n"):
        acc2 = 100 * sg["progressing"] // sg["n"]
        print(f"CONTROL B (known healthy ShareGPT): judge called {acc2}% 'progressing'.")
        print("  >=85% = recognises healthy; low = judge is OVER-STRICT, so the low")
        print("  healthy-yield on self-talk would be an artifact, not a real finding.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
