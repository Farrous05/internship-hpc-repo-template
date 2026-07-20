"""Package a finished 10k harvest and email the compact artifacts.

Run once AFTER the (6 parallel) harvest jobs end -- via a thin slurm wrapper
submitted with --dependency=afterany on them. The raw runs (~1GB for 10k) are
too big to email (the local relay caps at ~10MB/message), so we build the
training-ready artifacts and mail THOSE:
  - method_a_v3.jsonl (+ heldout)  the dataset
  - positive_seeds.json            option-B: seeds that produced positives
  - summary.txt / build.log        detection rate, onset spread, build output

NOTE the negatives in this snapshot are PLACEHOLDER (old seed pool). The
definitive negatives (option B, subjects matched to the real positives) are
generated afterwards and the dataset rebuilt. This is a preservation snapshot.

Usage:
  python scripts/package_harvest.py \
      --harvest-dir /dais/fs/scratch/fash/results/qwen10k \
      --topics /u/fash/babel-ai/data/topics_10k.json \
      --sharegpt /u/fash/babel-ai/data/sharegpt_real.json \
      --neg-pool data/healthy_generated.json \
      --mailto faresshretah@gmail.com
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import smtplib
import statistics
import subprocess
import sys
import time
from email.message import EmailMessage

HERE = os.path.dirname(os.path.abspath(__file__))
PART_BYTES = 9 * 1024 * 1024   # keep each email under the ~10MB relay cap


def summarize(harvest_dir: str, out_path: str) -> str:
    metas = glob.glob(os.path.join(harvest_dir, "**", "*_meta.json"),
                      recursive=True)
    onsets, with_id = [], 0
    for m in metas:
        try:
            d = json.load(open(m))
        except Exception:
            continue
        onsets.append(d.get("collapse_onset_round"))
        if d.get("seed_id"):
            with_id += 1
    det = [o for o in onsets if o is not None]
    n = len(metas)
    lines = [
        f"harvest runs written : {n}",
        f"detected collapse     : {len(det)}/{n} = "
        f"{100 * len(det) / max(n, 1):.0f}%",
    ]
    if det:
        lines.append(
            f"onset round median/max: {statistics.median(det):.0f} / {max(det)}"
        )
    lines.append(f"runs with seed_id     : {with_id}/{n} (option-B provenance)")
    text = "\n".join(lines)
    open(out_path, "w").write(text + "\n")
    print(text, flush=True)
    return text


def run(cmd: list[str], log_path: str) -> bool:
    """Run a subprocess, tee stderr to log_path; return success (never raise)."""
    print("+ " + " ".join(cmd), flush=True)
    with open(log_path, "a") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return p.returncode == 0


def make_zip(work: str, stamp: str, members: list[str]) -> str:
    import zipfile
    zip_path = os.path.join(work, f"harvest_{stamp}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for m in members:
            full = os.path.join(work, m)
            if os.path.exists(full):
                z.write(full, m)
    return zip_path


def split(zip_path: str) -> list[str]:
    parts, data = [], open(zip_path, "rb").read()
    if len(data) <= PART_BYTES:
        return [zip_path]
    for i in range(0, len(data), PART_BYTES):
        pp = f"{zip_path}.part{i // PART_BYTES:02d}"
        open(pp, "wb").write(data[i:i + PART_BYTES])
        parts.append(pp)
    return parts


def mail_part(mailto: str, subject: str, body: str, attach: str) -> bool:
    msg = EmailMessage()
    msg["From"] = f"{os.getenv('USER', 'harvest')}@localhost"
    msg["To"] = mailto
    msg["Subject"] = subject
    msg.set_content(body)
    with open(attach, "rb") as f:
        msg.add_attachment(f.read(), maintype="application",
                           subtype="octet-stream",
                           filename=os.path.basename(attach))
    try:
        with smtplib.SMTP("localhost", 25, timeout=60) as s:
            s.send_message(msg)
        return True
    except Exception as e:
        print(f"  mail failed: {e}", flush=True)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest-dir", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--neg-pool", default="data/healthy_generated.json")
    ap.add_argument("--embedder", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--mailto", default="faresshretah@gmail.com")
    ap.add_argument("--work", default=None)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M")
    work = args.work or f"/dais/fs/scratch/{os.getenv('USER')}/package_{stamp}"
    os.makedirs(work, exist_ok=True)
    build_log = os.path.join(work, "build.log")
    ds = os.path.join(work, "method_a_v3.jsonl")

    summarize(args.harvest_dir, os.path.join(work, "summary.txt"))

    # build the dataset (positives + placeholder negatives + sharegpt retention)
    run([sys.executable, os.path.join(HERE, "build_offline_dataset.py"),
         args.harvest_dir, args.topics, ds,
         "--embedder", args.embedder,
         "--negatives-pool", args.neg_pool,
         "--sharegpt", args.sharegpt], build_log)

    # option-B seed list (which subjects produced positives)
    run([sys.executable, os.path.join(HERE, "extract_positive_seeds.py"),
         args.harvest_dir, os.path.join(work, "positive_seeds.json")],
        build_log)

    zip_path = make_zip(work, stamp, [
        "summary.txt", "build.log", "method_a_v3.jsonl",
        "method_a_v3_heldout.jsonl", "positive_seeds.json",
    ])
    parts = split(zip_path)
    summary = open(os.path.join(work, "summary.txt")).read()
    zname = os.path.basename(zip_path)
    print(f"zip {os.path.getsize(zip_path) / 1e6:.1f} MB -> {len(parts)} "
          f"part(s), mailing to {args.mailto}", flush=True)

    for i, p in enumerate(parts, 1):
        body = (f"10k harvest package ({stamp}), part {i}/{len(parts)}.\n\n"
                f"Reassemble ALL parts:  cat {zname}.part* > {zname}  "
                f"then unzip.\n\n=== summary ===\n{summary}")
        ok = mail_part(args.mailto, f"harvest package {stamp} - part "
                       f"{i}/{len(parts)}", body, p)
        print(f"  part {i}/{len(parts)}: {'sent' if ok else 'FAILED'}",
              flush=True)
    print(f"done -> {work}", flush=True)


if __name__ == "__main__":
    main()
