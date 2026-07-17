"""Shared LLM-judge: is a conversation window PROGRESSING or LOOPING?

Metrics (word-overlap, cosine) conflate "same topic" (healthy) with "same content"
(collapse), so they cannot pick healthy LLM-to-LLM windows. This asks the semantic
question instead and gives a detector-INDEPENDENT label.

VALIDATED 2026-07-16: with model=gpt-4o this prompt called 97% of known
collapse-onset windows "looping" (control). gpt-4o-mini scored only 70% -- it
mistook surface variety ("new ingredients mentioned") for progress and passed
obvious mutual-appreciation loops. Do NOT downgrade the model without re-running
the control in scripts/judge_windows.py.
"""
from __future__ import annotations

import json
import random
import time

TURN_CAP = 500  # match the dataset's history cap: judge what we actually feed

PROMPT = """You are analyzing consecutive turns of a conversation between two AI assistants.

Decide: is this conversation PROGRESSING or LOOPING?

LOOPING = the conversation is stuck. It is not going anywhere and could continue
like this forever without ever resolving or building anything.

CRITICAL -- the most common mistake is calling a loop "progressing" because new
nouns or details appear. SURFACE VARIETY IS NOT PROGRESS. A conversation is LOOPING
if every turn makes the SAME conversational move, even when the details differ.
Watch for these patterns, which are LOOPING:
  * Mutual appreciation: "You're absolutely right!", "I'm glad you're excited...",
    "I'm so glad you brought up..." -- repeated every single turn, each time with a
    slightly different detail attached. This is still a loop.
  * Endless polishing: "Here's a refined / polished / more comprehensive version..."
    turn after turn.
  * Each turn complimenting or restating the previous turn and appending one new
    token detail, without anything being answered, decided, or built.

PROGRESSING = the conversation actually advances: a question gets answered, a problem
gets diagnosed or fixed, a decision is reached, or substantive new content is
introduced AND built upon. Staying on ONE TOPIC is normal and still PROGRESSING.

EQUALLY IMPORTANT -- do NOT confuse REPEATED PHRASING with a loop:
  * If one speaker keeps making similar-sounding REQUESTS ("write that scene",
    "give me a description of X", "now do the same for Y") but each turn PRODUCES
    NEW CONTENT -- a different scene, a new answer, a new section -- that is
    PROGRESSING. Real task-driven conversations look exactly like this.
  * Someone asking for a variation, a rewrite, or the next step is DIRECTING the
    work, not looping.
  * A LOOP is when NOTHING NEW IS PRODUCED: the speakers restate, compliment, or
    re-polish the SAME material.

THE TEST, in order:
  1. Is new content actually being produced each turn? If YES -> PROGRESSING, even
     if the requests are worded similarly.
  2. If nothing new is being produced, and the turns just agree with / re-polish
     each other in slightly different words -> LOOPING.

Conversation:
{convo}

Respond with JSON only:
{{"verdict": "progressing" | "looping", "confidence": 0.0-1.0, "reason": "<one short sentence>"}}"""


def fmt(turns):
    return "\n\n".join(f"Turn {i+1}: {t[:TURN_CAP]}" for i, t in enumerate(turns))


def judge(client, model, turns, retries=6):
    """-> {"verdict": "progressing"|"looping"|"error", "confidence": float, "reason": str}

    Retries with exponential backoff: concurrent sweeps hit OpenAI rate limits, and
    a swallowed 429 would silently look like "not healthy" and corrupt the yield.
    An error is only returned after every retry is exhausted.
    """
    last = ""
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": PROMPT.format(convo=fmt(turns))}],
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:
            last = str(e)
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt + random.random(), 30))
    return {"verdict": "error", "confidence": 0.0, "reason": last[:160]}


def is_healthy(client, model, turns):
    return judge(client, model, turns).get("verdict") == "progressing"
