"""Judge-verified healthy negatives -> a reusable pool for the dataset builder.

Metric thresholds cannot separate "healthy on-topic exchange" from "loop with
surface variety", so negatives are SELECTED BY THE JUDGE, not by a threshold.
We sweep EVERY pre-onset window of every run (no margin heuristic -- the judge is
the filter) and keep whatever it certifies as progressing.

Each window is tagged with `seed_frac`: the fraction of its turns that came from
the fetcher's ShareGPT seed rather than the model. This matters because a run's
opening is mostly seed, so a "healthy" window there is really healthy HUMAN chat
with a model turn appended. seed_frac == 0 means genuine healthy LLM-to-LLM
self-talk -- the thing we actually want, and the thing worth counting.

Judging runs on the CAPPED text, i.e. exactly what training will see.

Usage:
  OPENAI_API_KEY=... python scripts/select_negatives.py RESULTS_DIRS \
      [--max-per-run 4] [--max-total 1200] [--model gpt-4o] [--workers 12]
"""
from __future__ import annotations

import argparse
import collections
import glob
import importlib.util
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

_here = os.path.dirname(__file__)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_b = _load("b", "build_offline_dataset.py")
_j = _load("j", "judge_lib.py")

RNG = random.Random(0)
W = 5  # history turns (matches W_CTX)


def candidates(meta_paths, max_per_run, max_total):
    out = []
    for m in meta_paths:
        if "BUGGYSEED" in m:
            continue
        run = _b.load_run(m)
        if not run or run["onset"] is None or run["recovered"]:
            continue
        o, c, seed = run["onset"], run["contents"], run["is_seed"]
        # every window whose TARGET lands strictly before onset
        ts = [t for t in range(W, min(o, len(c)))]
        RNG.shuffle(ts)
        for t in ts[:max_per_run]:
            hist, tgt = c[t - W:t], c[t]
            if not tgt.strip() or any(not h.strip() for h in hist):
                continue
            out.append({
                "messages": _b.window_to_messages(hist),
                "target": {"role": "assistant",
                           "content": _b.cap_target(tgt, _b.NEG_TARGET_CAP)},
                "label_type": "negative", "source": "babel_healthy",
                "run_id": run["run_id"], "src_model": run["src_model"],
                "target_iter": t, "onset_round": o,
                "seed_frac": round(sum(seed[t - W:t + 1]) / (W + 1), 2),
            })
    RNG.shuffle(out)
    return out[:max_total]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dirs", help="comma-separated results dirs")
    ap.add_argument("--max-per-run", type=int, default=4)
    ap.add_argument("--max-total", type=int, default=1200)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="data/negatives_pool.json")
    args = ap.parse_args()

    metas = []
    for rd in args.results_dirs.split(","):
        metas += glob.glob(os.path.join(rd.strip(), "**", "*_meta.json"),
                           recursive=True)
    cands = candidates(sorted(set(metas)), args.max_per_run, args.max_total)
    print(f"candidate pre-onset windows: {len(cands)}", flush=True)

    from openai import OpenAI
    client = OpenAI()

    def run_one(row):
        turns = [m["content"] for m in row["messages"]] + [row["target"]["content"]]
        v = _j.judge(client, args.model, turns)
        row["judge_verdict"] = v.get("verdict")
        row["judge_reason"] = v.get("reason")
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        judged = list(ex.map(run_one, cands))

    kept = [r for r in judged if r["judge_verdict"] == "progressing"]
    errs = [r for r in judged if r["judge_verdict"] == "error"]
    valid = [r for r in judged if r["judge_verdict"] in ("progressing", "looping")]
    json.dump({"n_judged": len(judged), "n_valid": len(valid), "n_kept": len(kept),
               "errors": len(errs), "negatives": kept}, open(args.out, "w"), indent=1)

    print("\n=== JUDGE-VERIFIED HEALTHY POOL ===")
    print(f"  sent {len(judged)} | valid verdicts {len(valid)} | ERRORS {len(errs)}")
    if errs:
        print(f"  !! first error: {errs[0]['judge_reason']}")
    if len(errs) > 0.05 * max(len(judged), 1):
        print("  !! >5% errors -- yields below are UNRELIABLE, fix and re-run.")
    # yields are over VALID verdicts only: an error must never look like "not healthy"
    print(f"  kept {len(kept)} / {len(valid)} valid "
          f"({100*len(kept)//max(len(valid),1)}% healthy)")

    def bucket(sf):
        return "pure LLM (no seed)" if sf == 0 else (
            "mostly LLM (<=33% seed)" if sf <= 0.34 else "seed-heavy (>33%)")
    for label in ["pure LLM (no seed)", "mostly LLM (<=33% seed)", "seed-heavy (>33%)"]:
        k = sum(1 for r in kept if bucket(r["seed_frac"]) == label)
        c = sum(1 for r in valid if bucket(r["seed_frac"]) == label)
        print(f"  {label:24s} kept {k:4d} / {c:4d} valid  "
              f"({100*k//max(c,1)}% healthy)")
    dist = collections.Counter(r["target_iter"] for r in kept)
    print("  kept by target iteration:", dict(sorted(dist.items())[:12]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
