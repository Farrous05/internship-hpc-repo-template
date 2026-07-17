"""Generate healthy (non-collapsed) two-voice conversations to use as negatives.

Why generate instead of mine: collapsed data is FREE (100% of self-loop runs
collapse) while healthy LLM-to-LLM data is the bottleneck (~10% of deep pre-onset
windows survive the judge). So we synthesize the half that is hard.

Why this beats the ShareGPT fallback: ShareGPT negatives carry real HUMAN turns
(short, imperative, typo-ridden) while the positives are two assistant voices --
a style tell the model can learn instead of "am I looping?". A generated
conversation can be two assistant voices that PROGRESS, matching the positives'
format so the ONLY difference is the looping property.

Why topics come from the SEEDS, not a hand-written list: the positives' topics are
whatever the ShareGPT seeds covered. A hand-picked domain list would make the two
classes differ in TOPIC as well as in looping -- handing the model a topic shortcut
(the same bug as Metal-vs-everything). Seeding healthy conversations from the SAME
ShareGPT pool the collapse runs use keeps one topic distribution across both
classes, so looping is the only systematic difference.

Every conversation is judge-verified with the control-validated instrument
(judge_lib: 97% on known loops / 90% on known-healthy), so "is this actually
healthy?" is measured, not assumed. Anything judged looping is discarded.

Note: Method A is cross-model by design (positives already span Qwen-7B,
Llama-70B, Qwen-72B, GPT-4o-mini), so the generator need not be the target model.
Vary --model across runs so one generator's voice never becomes a cue.

Usage:
  OPENAI_API_KEY=... python scripts/generate_healthy.py --n 200 \
      [--seeds /u/fash/babel-ai/data/sharegpt_real.json] [--model gpt-4o]
      [--turns 6] [--workers 4] [--out data/healthy_generated.json]
"""
from __future__ import annotations

import argparse
import collections
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
STYLES = [
    "concise and practical", "detailed and thorough", "curious and exploratory",
    "step-by-step and methodical", "analytical, weighing trade-offs",
]

GEN_PROMPT = """Two AI assistants are having a conversation that starts from this opening:

--- OPENING ---
{opening}
--- END OPENING ---

Write a realistic {turns}-turn conversation between the two assistants continuing from
that opening. Stay on the opening's subject.

HARD REQUIREMENTS:
1. The conversation must PROGRESS. Every single turn must add something NEW: answer a
   question, supply a fact, diagnose a problem, make a decision, or move to the next
   step. By the final turn something must actually have been resolved or built.
2. NEVER have a turn compliment, praise, agree with, restate, summarise or "polish"
   the previous turn. Specifically FORBIDDEN openings: "You're absolutely right",
   "I'm glad you...", "Great point", "Certainly! Here's a refined/polished version",
   "That's a comprehensive overview". A turn that mainly reacts to the previous turn
   is a failure.
3. Both speakers talk like helpful assistants (assistant-to-assistant, not a human
   chatting). They may disagree, ask each other questions, and correct one another.
4. Each turn is substantive: roughly 60-140 words. Style: {style}.
5. Be concrete -- real specifics, numbers, names -- not generic filler.

Respond with JSON only:
{{"turns": ["turn 1 text", "turn 2 text", ...]}}  exactly {turns} turns."""


def load_openings(path):
    """First human turn of each ShareGPT conversation = the same topic pool the
    collapse runs are seeded from."""
    out = []
    for conv in json.load(open(path)):
        for m in conv.get("items") or []:
            if m.get("from") == "human" and (m.get("value") or "").strip():
                out.append(m["value"][:800])
                break
    return out


def generate_one(client, model, turns, opening):
    style = RNG.choice(STYLES)
    try:
        r = client.chat.completions.create(
            model=model, temperature=1.0,   # variety across conversations
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": GEN_PROMPT.format(
                turns=turns, opening=opening, style=style)}],
        )
        data = json.loads(r.choices[0].message.content)
        t = [x for x in data.get("turns", []) if isinstance(x, str) and x.strip()]
        return t if len(t) >= turns else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seeds", default="/u/fash/babel-ai/data/sharegpt_real.json",
                    help="topic pool -- the SAME seeds the collapse runs use")
    ap.add_argument("--model", default="llama3.3:70b-instruct-q8_0",
                    help="GENERATOR (local Ollama tag). Must differ from --judge-model")
    ap.add_argument("--judge-model", default="gemma4:31b-it-bf16",
                    help="JUDGE (control-validated: gemma4:31b = 95%%/95%%). Never "
                         "the same as --model: a model grading its own output "
                         "favours it (self-preference), so the verification would "
                         "be worthless.")
    ap.add_argument("--turns", type=int, default=6, help="5 history + 1 target")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/healthy_generated.json")
    args = ap.parse_args()

    if args.model == args.judge_model:
        ap.error("--model and --judge-model must differ: a model grading its own "
                 "output favours it, so the healthy-verification would be void.")
    openings = load_openings(args.seeds)
    print(f"topic pool: {len(openings)} real ShareGPT openings", flush=True)
    print(f"generator: {args.model} | judge: {args.judge_model}", flush=True)

    from openai import OpenAI
    client = OpenAI()

    def one(i):
        opening = openings[i % len(openings)] if openings else "a technical question"
        turns = generate_one(client, args.model, args.turns, opening)
        if not turns:
            return None
        hist, tgt = turns[:args.turns - 1], turns[args.turns - 1]
        # judge the SAME text training will see (capped), incl. the target turn
        v = _j.judge(client, args.judge_model, turns[:args.turns])
        return {
            "messages": _b.window_to_messages(hist),
            "target": {"role": "assistant",
                       "content": _b.cap_target(tgt, _b.NEG_TARGET_CAP)},
            "label_type": "negative", "source": "generated_healthy",
            "run_id": "gen:%s:%d" % (args.model, i),
            "src_model": "generated:" + args.model,
            "opening": opening[:120],
            "judge_verdict": v.get("verdict"), "judge_reason": v.get("reason"),
        }

    order = list(range(len(openings) or 1))
    RNG.shuffle(order)
    idx = [order[i % len(order)] for i in range(args.n)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for r in ex.map(one, idx) if r]

    kept = [r for r in rows if r["judge_verdict"] == "progressing"]
    errs = sum(1 for r in rows if r["judge_verdict"] == "error")
    valid = [r for r in rows if r["judge_verdict"] in ("progressing", "looping")]
    json.dump({"n_generated": len(rows), "n_valid": len(valid), "n_kept": len(kept),
               "errors": errs, "negatives": kept}, open(args.out, "w"), indent=1)

    print("=== GENERATED HEALTHY NEGATIVES ===")
    print(f"  requested {args.n} | generated {len(rows)} | valid verdicts {len(valid)}"
          f" | errors {errs}")
    print(f"  judge-verified healthy: {len(kept)} / {len(valid)} "
          f"({100*len(kept)//max(len(valid),1)}%)")
    print(f"  distinct seed topics used: "
          f"{len(collections.Counter(r['opening'] for r in kept))}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
