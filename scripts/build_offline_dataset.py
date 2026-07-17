"""Build the offline SFT dataset for the <end>-token study.

Every example has the SAME shape: a conversation history (``messages``) plus the
next assistant turn we want the model to produce (``target``). The model learns
to make ``target`` start with ``<end>`` exactly when the conversation has
collapsed, and to continue normally when it has not.

  * POSITIVE  — history is a window ending at the collapse onset (the loop is
    visible); target = "<end> " + a topic from the bank that is far (in meaning)
    from what the run collapsed on.
  * NEGATIVE (in-distribution) — an early healthy window from the same run;
    target = the real next healthy turn, no <end>.
  * NEGATIVE (sharegpt) — a real chat history; target = its real reply. Added
    only if a ShareGPT file is supplied (retention bucket).

Self-loop turns are one agent's outputs; we assign alternating user/assistant
roles ending in ``user`` (matching how babel-ai fed them back), so the target is
the assistant's response.

Topic grafting is distance-aware when an embedder is given (``--embedder`` = a
local HF SentenceTransformer id); otherwise it falls back to random selection
(fine for a plumbing dry-run). See Dataset_spec.md.

Usage:
  python scripts/build_offline_dataset.py RESULTS_DIR TOPIC_BANK OUT.jsonl \
      [--embedder BAAI/bge-large-en-v1.5] [--sharegpt sharegpt.json]
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detector.collapse import CollapseDetector  # noqa: E402

csv.field_size_limit(10**7)
_WORD_RE = re.compile(r"\w+")

# ---- knobs (Dataset_spec.md §9) --------------------------------------------
W_CTX = 5              # turns of history per example (enough to show a loop)
TURN_CHAR_CAP = 500    # cap each *history* turn to this many chars so long looped
#                        paragraphs don't push the <end> target past max_length.
#                        Targets are never capped (they are what the model emits).
POS_QUALITY = 0.70     # keep positives whose peak windowed sim near onset >= this
TOPIC_DIST_CUTOFF = 0.55   # min meaning-distance topic<->collapsed centroid
TOPIC_LEX_MAX = 0.30       # drop topics sharing > this word-overlap fraction
# ---- in-distribution (self-talk) negatives: clean, deep, far from the loop ---
# C1: onset is usually pinned at the detector's warmup floor (10), so the loop
# starts BEFORE onset and the pre-onset region is contaminated. Only draw healthy
# self-talk negatives from runs that collapse LATE (a real healthy stretch exists),
# gate the target turn on DIRECT word-overlap with the loop (not the diluted
# windowed mean), and keep it deep in the conversation (I1: no "opening = healthy"
# shortcut). C2: cap the negative target so it can't dominate the loss.
NEG_MIN_ONSET = 16     # only take self-talk negatives from runs with onset >= this
NEG_MARGIN = 6         # target turn must be >= this many turns before onset
NEG_MIN_DEPTH = 6      # target turn must be at conversation depth >= this (I1)
NEG_LOOP_OVERLAP_MAX = 0.30  # target's word-overlap with loop turns must be < this
NEG_TARGET_CAP = 800   # cap the negative target turn to this many chars (C2)
NEG_PER_RUN = 2        # healthy (negative) examples sampled per run
SHAREGPT_RATIO = 0.5   # sharegpt negatives sampled as this fraction of #positives
SHAREGPT_MIN_DEPTH = 3     # cut ShareGPT at turn >= this (I1: not just openings)
SHAREGPT_MAX_CHARS = 1500  # skip sharegpt turns longer than this (keep examples clean)
HELD_OUT_FRAC = 0.10
RNG = random.Random(0)


# ---- run parsing ------------------------------------------------------------
def load_run(meta_path: str):
    """Load a run indexed by `iteration`.

    IMPORTANT: `iteration` and `collapse_onset_round` count EVERY row, including
    the fetcher's seed messages (1-5 per run, no agent_id). Dropping the seed rows
    and then indexing the remainder shifted every window LATER by that run's seed
    length -- which silently pulled "pre-onset healthy" negatives into the loop.
    So we keep all rows and place each at its own `iteration`.
    """
    meta = json.load(open(meta_path))
    onset = meta.get("collapse_onset_round")
    csvs = glob.glob(os.path.join(os.path.dirname(meta_path), "*.csv"))
    if not csvs:
        return None
    by_iter = {}
    with open(csvs[0]) as f:
        for row in csv.DictReader(f):
            try:
                it = int(row["iteration"])
            except (TypeError, ValueError):
                continue
            try:
                an = ast.literal_eval(row["analysis"]) if row["analysis"] else {}
            except (ValueError, SyntaxError):
                an = {}
            by_iter[it] = (row["content"],
                           an.get("semantic_similarity_window"),
                           not (row.get("agent_id") or "").strip())
    if not by_iter:
        return None
    n = max(by_iter) + 1
    contents = [by_iter.get(i, ("", None, False))[0] for i in range(n)]
    windows = [by_iter.get(i, ("", None, False))[1] for i in range(n)]
    is_seed = [by_iter.get(i, ("", None, False))[2] for i in range(n)]
    cfg = (meta.get("config") or {}).get("agent_configs") or [{}]
    src_model = cfg[0].get("model", "?")
    run_id = os.path.basename(os.path.dirname(meta_path))
    recovered = bool((meta.get("recovery") or {}).get("recovered"))
    return dict(onset=onset, contents=contents, windows=windows,
                is_seed=is_seed, src_model=src_model, run_id=run_id,
                recovered=recovered)


def collapse_mode(windows, onset):
    """Peak windowed similarity in [onset, onset+3] -> quality/mode, or None."""
    seg = [w for w in windows[onset:onset + 4] if isinstance(w, (int, float))]
    if not seg:
        return None
    peak = max(seg)
    if peak >= 0.90:
        return "verbatim", peak
    if peak >= POS_QUALITY:
        return "paraphrase", peak
    return None  # borderline -> drop


# ---- windows -> messages ----------------------------------------------------
def window_to_messages(contents):
    """Alternating user/assistant over a window, ending in 'user'."""
    W = len(contents)
    return [
        {"role": "user" if (W - 1 - i) % 2 == 0 else "assistant",
         "content": c[:TURN_CHAR_CAP]}
        for i, c in enumerate(contents)
    ]


# ---- topic selection --------------------------------------------------------
def _words(t):
    return set(_WORD_RE.findall(t.lower()))


def _lex_overlap(a, b):
    sa, sb = _words(a), _words(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class OpenAIEmbedder:
    """SBERT-compatible embedder backed by OpenAI text-embedding-3-large.

    Exposes ``encode(texts, convert_to_tensor=True)`` returning a torch tensor
    (1-D for a str, 2-D for a list) so it drops straight into TopicPicker's
    ``cos_sim`` / ``.mean(dim=0)`` calls. Reads OPENAI_API_KEY from the env.
    Used for topic-distance grafting now that compute/login nodes have internet
    (validated 2026-07-15) -- a stronger, long-context embedder than MiniLM for
    measuring how *far* a graft topic is from the collapsed content.
    """

    def __init__(self, model="text-embedding-3-large"):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI()          # OPENAI_API_KEY from env
        self.cache = {}

    def encode(self, texts, convert_to_tensor=True, **_):
        import torch
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        norm = [((t if t and t.strip() else " ")[:30000]) for t in items]
        missing = [t for t in dict.fromkeys(norm) if t not in self.cache]
        for i in range(0, len(missing), 256):
            batch = missing[i:i + 256]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            for t, e in zip(batch, resp.data):
                self.cache[t] = e.embedding
        out = torch.tensor([self.cache[t] for t in norm])
        return out[0] if single else out


class TopicPicker:
    def __init__(self, topics, embedder=None):
        self.topics = topics
        self.embedder = embedder
        self.topic_emb = None
        if embedder is not None:
            self.topic_emb = embedder.encode(topics, convert_to_tensor=True)

    def pick(self, collapsed_texts):
        """A topic far (in meaning) from what the run collapsed on."""
        if self.embedder is None:
            return RNG.choice(self.topics)  # dry-run fallback
        from sentence_transformers.util import cos_sim
        centroid = self.embedder.encode(
            collapsed_texts, convert_to_tensor=True).mean(dim=0)
        eligible = []
        for i, t in enumerate(self.topics):
            dist = 1.0 - float(cos_sim(self.topic_emb[i], centroid).item())
            lex = max(_lex_overlap(t, c) for c in collapsed_texts)
            if dist >= TOPIC_DIST_CUTOFF and lex <= TOPIC_LEX_MAX:
                eligible.append((dist, t))
        if eligible:
            return RNG.choice(eligible)[1]
        # nothing clears the bar -> take the single farthest topic
        dists = [(1.0 - float(cos_sim(self.topic_emb[i], centroid).item()), t)
                 for i, t in enumerate(self.topics)]
        return max(dists)[1]


# ---- example construction ---------------------------------------------------
def cap_target(text, cap):
    """Cap a target turn near `cap` chars, preferring a sentence boundary so we
    don't teach the model to stop mid-word (C2)."""
    if len(text) <= cap:
        return text
    head = text[:cap]
    end = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    return head[:end + 1] if end >= cap // 2 else head


def build_positive(run, picker):
    if run["recovered"]:
        return None                    # detector+recovery disagree -> drop (suspect)
    onset = run["onset"]
    lo = max(0, onset - W_CTX + 1)
    window = run["contents"][lo:onset + 1]
    if len(window) < 2:
        return None
    topic = picker.pick(window)
    return {
        "messages": window_to_messages(window),
        "target": {"role": "assistant", "content": "<end> " + topic},
        "label_type": "positive", "source": "babel",
        "run_id": run["run_id"], "onset_round": onset,
        "src_model": run["src_model"],
    }


def build_negatives(run):
    """Clean, deep, in-distribution (self-talk) healthy examples.

    C1: only from LATE-collapsing runs (onset >= NEG_MIN_ONSET) where a real
    healthy stretch exists; the target turn must be far from onset (NEG_MARGIN),
    deep in the conversation (NEG_MIN_DEPTH, kills the position shortcut I1), and
    have low DIRECT word-overlap with the loop turns [onset, onset+3] -- not the
    diluted windowed mean that let looping turns through. C2: cap the target.
    """
    onset, contents = run["onset"], run["contents"]
    if onset is None or onset < NEG_MIN_ONSET:
        return []
    loop_turns = contents[onset:onset + 4]
    if not loop_turns:
        return []
    hi = onset - NEG_MARGIN                # target turn must be this far pre-onset
    lo = max(W_CTX, NEG_MIN_DEPTH)         # ...and this deep in the conversation
    candidates = list(range(lo, hi))
    RNG.shuffle(candidates)
    out = []
    for tgt in candidates:
        overlap = max(_lex_overlap(contents[tgt], lt) for lt in loop_turns)
        if overlap >= NEG_LOOP_OVERLAP_MAX:
            continue                        # already resembles the loop -> skip
        window = contents[tgt - W_CTX:tgt]
        if len(window) < 2:
            continue
        out.append({
            "messages": window_to_messages(window),
            "target": {"role": "assistant",
                       "content": cap_target(contents[tgt], NEG_TARGET_CAP)},
            "label_type": "negative", "source": "babel_preonset",
            "run_id": run["run_id"], "src_model": run["src_model"],
        })
        if len(out) >= NEG_PER_RUN:
            break
    return out


_SGPT_ROLE = {"human": "user", "gpt": "assistant"}


def build_sharegpt_negatives(sharegpt_path, n):
    """Retention negatives from real human<->assistant chat (ShareGPT).

    Cut at each assistant ("gpt") turn preceded by a user ("human") turn:
    history = up to the last W_CTX turns before it, target = that reply.
    Real roles are kept (this is genuine, non-collapsed dialogue). Sample n.
    """
    data = json.load(open(sharegpt_path))
    cands = []
    for conv in data:
        items = conv.get("items") or []
        for i in range(1, len(items)):
            if i < SHAREGPT_MIN_DEPTH:
                continue                       # I1: cut deep, not at openings
            if items[i].get("from") != "gpt":
                continue
            if items[i - 1].get("from") != "human":
                continue                       # need a clean user->assistant cut
            tgt = items[i].get("value") or ""
            if not tgt or len(tgt) > SHAREGPT_MAX_CHARS:
                continue
            hist = items[max(0, i - W_CTX):i]
            msgs = [{"role": _SGPT_ROLE[m["from"]],
                     "content": (m.get("value") or "")[:TURN_CHAR_CAP]}
                    for m in hist if m.get("from") in _SGPT_ROLE]
            if not msgs or msgs[-1]["role"] != "user":
                continue
            if any(len(m["content"]) > SHAREGPT_MAX_CHARS for m in msgs):
                continue
            cands.append({
                "messages": msgs,
                "target": {"role": "assistant",
                           "content": cap_target(tgt, NEG_TARGET_CAP)},
                "label_type": "negative", "source": "sharegpt",
                "run_id": "sharegpt:" + str(conv.get("id", "?")),
                "src_model": "sharegpt",
            })
    RNG.shuffle(cands)
    return cands[:n]


# ---- driver -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir",
                    help="one dir, or a comma-separated list (borrowed,harvest)")
    ap.add_argument("topic_bank")
    ap.add_argument("out")
    ap.add_argument("--embedder", default=None,
                    help="local HF SentenceTransformer id; omit for random")
    ap.add_argument("--openai-embedder", default=None,
                    help="OpenAI embedding model id, e.g. text-embedding-3-large "
                         "(needs OPENAI_API_KEY + internet); overrides --embedder")
    ap.add_argument("--sharegpt", default=None)
    ap.add_argument("--negatives-pool", default=None,
                    help="judge-verified healthy negatives (generate_healthy.py). "
                         "REPLACES the mined pre-onset negatives: mining Qwen's own "
                         "self-talk yields only ~10% genuinely-healthy windows "
                         "(measured), whereas generated ones run ~91% and match the "
                         "positives' two-assistant format.")
    ap.add_argument("--max-negatives", type=int, default=None,
                    help="cap negatives. Watch the LOSS MASS, not the count: "
                         "negative targets are ~800 chars vs a positive's short "
                         "'<end> + topic', so too many negatives silently make the "
                         "model spend its training writing healthy chatter instead "
                         "of learning the trigger (this was the 71%% bug).")
    args = ap.parse_args()

    topics = [e["items"][0]["value"]
              for e in json.load(open(args.topic_bank))]
    embedder = None
    if args.openai_embedder:
        embedder = OpenAIEmbedder(args.openai_embedder)
    elif args.embedder:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(args.embedder)
    picker = TopicPicker(topics, embedder)

    rows, stats = [], dict(runs=0, excluded_buggyseed=0, dropped_borderline=0,
                           dropped_recovered=0, pos=0,
                           neg_preonset=0, neg_sharegpt=0)
    metas = []
    for rd in args.results_dir.split(","):
        metas += glob.glob(os.path.join(rd.strip(), "**", "*_meta.json"),
                           recursive=True)
    for m in sorted(set(metas)):
        if "BUGGYSEED" in m:                    # known-bad seed batch -> exclude
            stats["excluded_buggyseed"] += 1
            continue
        run = load_run(m)
        if not run or run["onset"] is None:
            continue
        stats["runs"] += 1
        mode = collapse_mode(run["windows"], run["onset"])
        if mode is None:
            stats["dropped_borderline"] += 1
            continue
        pos = build_positive(run, picker)
        if pos:
            pos["collapse_mode"] = mode[0]
            rows.append(pos)
            stats["pos"] += 1
        elif run["recovered"]:
            stats["dropped_recovered"] += 1
        if not args.negatives_pool:      # pool supersedes mined negatives
            negs = build_negatives(run)
            rows.extend(negs)
            stats["neg_preonset"] += len(negs)

    if args.negatives_pool:
        pool = json.load(open(args.negatives_pool))
        negs = pool.get("negatives", pool) if isinstance(pool, dict) else pool
        # never trust a pool row that wasn't judged healthy
        negs = [n for n in negs if n.get("judge_verdict", "progressing") == "progressing"]
        RNG.shuffle(negs)
        if args.max_negatives:
            negs = negs[:args.max_negatives]
        for n in negs:
            n.pop("judge_reason", None)
            rows.append(n)
        stats["neg_generated"] = len(negs)

    # retention negatives from ShareGPT, sized relative to #positives
    if args.sharegpt:
        n_sgpt = round(SHAREGPT_RATIO * stats["pos"])
        sgpt = build_sharegpt_negatives(args.sharegpt, n_sgpt)
        rows.extend(sgpt)
        stats["neg_sharegpt"] = len(sgpt)

    # drop exact-duplicate examples
    seen, deduped = set(), []
    for r in rows:
        k = json.dumps([r["messages"], r["target"]], sort_keys=True)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    stats["dropped_duplicates"] = len(rows) - len(deduped)
    rows = deduped

    # split by BASE run so no run leaks across train/held-out. ShareGPT chunk
    # ids (X_0, X_23 = chunks of one conversation) collapse to their base X.
    def split_key(run_id):
        if run_id.startswith("sharegpt:"):
            return "sharegpt:" + re.sub(r"_\d+$", "", run_id.split(":", 1)[1])
        return run_id
    keys = sorted({split_key(r["run_id"]) for r in rows})
    RNG.shuffle(keys)
    n_hold = max(1, int(len(keys) * HELD_OUT_FRAC))
    held = set(keys[:n_hold])
    with open(args.out, "w") as ftr, \
            open(args.out.replace(".jsonl", "_heldout.jsonl"), "w") as fho:
        for r in rows:
            (fho if split_key(r["run_id"]) in held else ftr).write(
                json.dumps(r) + "\n")

    print(json.dumps(stats, indent=2))
    print(f"held-out base-runs: {len(held)} / {len(keys)}")
    print(f"wrote {args.out} (+ _heldout.jsonl)")


if __name__ == "__main__":
    main()
