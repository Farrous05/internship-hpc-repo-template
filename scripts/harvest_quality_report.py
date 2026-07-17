"""Reliability report for a harvest results dir: is each collapse real, sustained,
and consistent with the (independent) recovery criterion? Prints an aggregate
verdict and flags weak runs, then dumps a hand-check sample so a human can judge
agreement (the gold standard). No embeddings recomputed -- uses the stored
semantic_similarity_window the detector already recorded.

Signals per run (all independent-ish evidence the loop is genuine):
  * peak_sim   -- max windowed similarity in [onset, onset+3]; the loop's strength
  * persist    -- how many of the next rounds STAY >= LOOP_SIM (a real loop holds,
                  a one-round blip does not)
  * recovered  -- recovery.recovered from meta: True means it escaped the loop,
                  which CONTRADICTS a sustained collapse (drop / suspect)
  * degenerate -- any looped turn empty or trivially short (error/echo, not content)

Usage: python scripts/harvest_quality_report.py RESULTS_DIR [n_sample]
"""
import ast
import csv
import glob
import json
import os
import random
import sys

csv.field_size_limit(10**7)

POS_QUALITY = 0.70   # builder's keep threshold (peak sim near onset)
LOOP_SIM = 0.70      # a round "counts as looping" at/above this windowed sim
VERBATIM = 0.90      # peak >= this == verbatim loop; else paraphrase
MIN_TURN_CHARS = 20  # a looped turn shorter than this is degenerate


def load(meta_path):
    d = json.load(open(meta_path))
    onset = d.get("collapse_onset_round")
    rec = (d.get("recovery") or {}).get("recovered")
    csvs = glob.glob(os.path.join(os.path.dirname(meta_path), "*.csv"))
    if not csvs or onset is None:
        return None
    contents, sims = [], []
    for r in csv.DictReader(open(csvs[0])):
        if not (r.get("agent_id") or "").strip():
            continue
        try:
            an = ast.literal_eval(r["analysis"]) if r["analysis"] else {}
        except (ValueError, SyntaxError):
            an = {}
        contents.append(r["content"])
        sims.append(an.get("semantic_similarity_window"))
    return dict(onset=onset, recovered=rec, contents=contents, sims=sims,
               run_id=os.path.basename(os.path.dirname(meta_path)))


def assess(run):
    onset, sims, contents = run["onset"], run["sims"], run["contents"]
    seg = [s for s in sims[onset:onset + 4] if isinstance(s, (int, float))]
    peak = max(seg) if seg else 0.0
    persist = sum(1 for s in sims[onset:] if isinstance(s, (int, float)) and s >= LOOP_SIM)
    loop_turns = contents[onset:onset + 4]
    degenerate = any(len(t.strip()) < MIN_TURN_CHARS for t in loop_turns) if loop_turns else True
    mode = "verbatim" if peak >= VERBATIM else ("paraphrase" if peak >= POS_QUALITY else "weak")
    return dict(peak=peak, persist=persist, degenerate=degenerate, mode=mode,
                recovered=run["recovered"])


def main():
    rdir = sys.argv[1]
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    runs = []
    for m in sorted(glob.glob(os.path.join(rdir, "**", "*_meta.json"), recursive=True)):
        r = load(m)
        if r:
            r["a"] = assess(r)
            runs.append(r)

    n = len(runs)
    keep = [r for r in runs if r["a"]["peak"] >= POS_QUALITY and not r["a"]["degenerate"]]
    verbatim = [r for r in runs if r["a"]["mode"] == "verbatim"]
    para = [r for r in runs if r["a"]["mode"] == "paraphrase"]
    weak = [r for r in runs if r["a"]["mode"] == "weak"]
    recovered = [r for r in runs if r["a"]["recovered"] is True]
    degen = [r for r in runs if r["a"]["degenerate"]]
    persists = sorted(r["a"]["persist"] for r in runs)

    print(f"=== harvest reliability report: {rdir} ===")
    print(f"runs: {n}")
    print(f"KEEP (peak>= {POS_QUALITY} & not degenerate): {len(keep)}  ({100*len(keep)//max(n,1)}%)")
    print(f"  mode: verbatim {len(verbatim)} | paraphrase {len(para)} | weak/drop {len(weak)}")
    print(f"persistence (rounds staying >= {LOOP_SIM} after onset): "
          f"min {persists[0]} median {persists[len(persists)//2]} max {persists[-1]}")
    print(f"recovery-consistency: recovered=True (suspect) {len(recovered)} | "
          f"stayed collapsed {n - len(recovered)}")
    print(f"degenerate (empty/echo loop turns): {len(degen)}")
    # peak-sim histogram
    import collections
    buckets = collections.Counter()
    for r in runs:
        p = r["a"]["peak"]
        b = "<0.70" if p < 0.70 else ("0.70-0.80" if p < 0.80 else ("0.80-0.90" if p < 0.90 else ">=0.90"))
        buckets[b] += 1
    print("peak-sim histogram:", dict(sorted(buckets.items())))

    # hand-check sample (mixed modes)
    random.seed(0)
    sample = random.sample(runs, min(n_sample, n))
    print(f"\n=== hand-check sample ({len(sample)} runs) ===")
    for r in sample:
        a = r["a"]
        print("-" * 72)
        print(f"{r['run_id'][:58]}")
        print(f"  onset={r['onset']} peak={a['peak']:.2f} mode={a['mode']} "
              f"persist={a['persist']} recovered={a['recovered']}")
        for i in range(max(0, r["onset"] - 1), min(len(r["contents"]), r["onset"] + 3)):
            s = r["sims"][i]
            ss = f"{s:.2f}" if isinstance(s, (int, float)) else "NA"
            print(f"    t{i:<2} sim={ss}: {' '.join(r['contents'][i].split())[:80]}")


if __name__ == "__main__":
    main()
