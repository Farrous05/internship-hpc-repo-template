"""Validate the vendored CollapseDetector against the borrowed 224-run corpus.

The recorded ``collapse_onset_round`` in each run's ``*_meta.json`` was produced
by babel-ai's detector *online*. Our vendored copy is a pure function of the
per-round ``semantic_similarity_window`` values, which are stored in the CSV — so
we can replay it offline (no model, no embeddings, no network) and confirm it
reproduces the recorded FIRST onset for every run.

We replay only up to the first onset (no rearm), which is well-defined: nothing
re-arms the detector before the first collapse, so injected runs reproduce too.

Usage:  python scripts/validate_detector.py <results_dir>
"""

from __future__ import annotations

import ast
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector.collapse import CollapseDetector  # noqa: E402

csv.field_size_limit(10**7)


def agent_windows(csv_path: str) -> list:
    """Per agent-turn windowed semantic similarity, in round order."""
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not (row.get("agent_id") or "").strip():
                continue  # skip fetcher seed turns
            try:
                an = ast.literal_eval(row["analysis"]) if row["analysis"] else {}
            except (ValueError, SyntaxError):
                an = {}
            out.append(an.get("semantic_similarity_window"))
    return out


def replay_first_onset(windows: list):
    det = CollapseDetector()
    for r, sw in enumerate(windows):
        det.update(round_idx=r, semantic_similarity_window=sw)
        if det.onset_round is not None:
            return det.onset_round
    return None


def main(results_dir: str) -> None:
    metas = sorted(glob.glob(os.path.join(results_dir, "**", "*_meta.json"),
                             recursive=True))
    match = mismatch = missing = 0
    disagreements = []
    for m in metas:
        recorded = json.load(open(m)).get("collapse_onset_round")
        siblings = glob.glob(os.path.join(os.path.dirname(m), "*.csv"))
        if not siblings:
            missing += 1
            continue
        replayed = replay_first_onset(agent_windows(siblings[0]))
        if replayed == recorded:
            match += 1
        else:
            mismatch += 1
            disagreements.append((os.path.basename(m), recorded, replayed))

    total = len(metas)
    print(f"runs:            {total}")
    print(f"onset reproduced: {match}")
    print(f"mismatches:       {mismatch}")
    print(f"missing csv:      {missing}")
    if total:
        print(f"agreement:        {100.0 * match / total:.1f}%")
    for name, rec, rep in disagreements[:20]:
        print(f"  MISMATCH recorded={rec} replayed={rep}  {name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
