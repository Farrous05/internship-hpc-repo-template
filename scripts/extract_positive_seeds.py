"""Option B: list the seed ids of harvest runs that become POSITIVES.

The healthy NEGATIVES must be seeded from the SAME subjects the positives came
from, or topic predicts the class (see Dataset_spec / the shortcut we hit). We
can only know which subjects actually produced positives AFTER the harvest, so
this runs between the harvest and the negatives generation:

    harvest -> extract_positive_seeds.py -> generate_healthy.py --seeds <this>

It reuses the builder's OWN eligibility test (load_run + collapse_mode +
build_positive), so the seed set matches exactly the runs the builder will turn
into positives -- not an approximation.

Usage:
  python scripts/extract_positive_seeds.py RESULTS_DIR OUT_SEEDS.json \
      --full-seed-pool /u/fash/babel-ai/data/seeds_10k.json
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_here = os.path.dirname(__file__)
_b = _load("b", os.path.join(_here, "build_offline_dataset.py"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", help="harvest output dir(s), comma-separated")
    ap.add_argument("out", help="JSON: the seed subset that produced positives")
    ap.add_argument(
        "--full-seed-pool",
        default="/u/fash/babel-ai/data/seeds_10k.json",
        help="the harvest's seed file; output is the subset of THIS by id",
    )
    args = ap.parse_args()

    # A no-op picker: build_positive only returns None for the reasons we care
    # about (recovered / window too short), never because of topic choice.
    dummy_picker = _b.TopicPicker([" "], embedder=None)

    metas = []
    for rd in args.results_dir.split(","):
        metas += glob.glob(
            os.path.join(rd.strip(), "**", "*_meta.json"), recursive=True
        )

    pos_seed_ids, no_provenance, runs, positives = set(), 0, 0, 0
    for m in sorted(set(metas)):
        if "BUGGYSEED" in m:
            continue
        run = _b.load_run(m)
        if not run or run["onset"] is None:
            continue
        runs += 1
        if _b.collapse_mode(run["windows"], run["onset"]) is None:
            continue
        if _b.build_positive(run, dummy_picker) is None:
            continue
        positives += 1
        # seed_id lives in the raw meta (added 2026-07-17); pre-fix runs lack it
        meta = json.load(open(m))
        sid = meta.get("seed_id")
        if sid is None:
            no_provenance += 1
        else:
            pos_seed_ids.add(sid)

    # Emit the subset of the full seed pool whose ids produced positives, in the
    # pool's own {id, items} schema so generate_healthy.py consumes it directly.
    pool = json.load(open(args.full_seed_pool))
    subset = [e for e in pool if e.get("id") in pos_seed_ids]
    json.dump(subset, open(args.out, "w"))

    print(f"runs with onset: {runs}  ->  positives: {positives}")
    print(f"distinct positive seed ids: {len(pos_seed_ids)}")
    print(f"matched in seed pool: {len(subset)}")
    if no_provenance:
        print(
            f"WARNING: {no_provenance} positives had NO seed_id (pre-provenance "
            f"runs). Re-harvest with the seed_id fix, or those subjects are lost."
        )
    print(f"wrote {len(subset)} seeds -> {args.out}")


if __name__ == "__main__":
    main()
