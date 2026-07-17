"""Lightweight, dependency-free language/script detection.

Vendored verbatim from babel-ai (``src/babel_ai/language.py``). Self-loops seeded
in English sometimes drift into another *script* (Latin -> CJK/Cyrillic/...),
which silently breaks the recovery guards (GPT-2 perplexity is English-only; the
not-parroting Jaccard check is against the English injection). A Unicode-block
tally catches a script switch with no model and no extra dependency.
"""

from __future__ import annotations

import unicodedata
from collections import Counter

_SCRIPT_KEYS = (
    "CJK",  # Chinese (and shared Han ideographs)
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",  # Korean
    "CYRILLIC",
    "ARABIC",
    "HEBREW",
    "DEVANAGARI",
    "GREEK",
    "LATIN",
)


def _script_of(ch: str) -> str | None:
    """Return the script key for a single alphabetic char, or None."""
    if not ch.isalpha():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    for key in _SCRIPT_KEYS:
        if key in name:
            return "JAPANESE" if key in ("HIRAGANA", "KATAKANA") else key
    return "OTHER"


def script_histogram(text: str) -> dict[str, float]:
    """Fraction of alphabetic characters in each script. Empty if no letters."""
    counts: Counter[str] = Counter()
    for ch in text:
        s = _script_of(ch)
        if s:
            counts[s] += 1
    total = sum(counts.values())
    if not total:
        return {}
    return {k: v / total for k, v in counts.items()}


def dominant_script(text: str) -> str:
    """The script of the majority of a turn's letters.

    Returns ``"NONE"`` when the text has no alphabetic characters -- callers
    should treat that as "no language signal", not as a switch.
    """
    hist = script_histogram(text)
    if not hist:
        return "NONE"
    return max(hist, key=hist.get)


def switched_language(text: str, baseline_script: str) -> bool:
    """True if ``text``'s dominant script differs from ``baseline_script``.

    ``NONE`` on either side is *not* a switch: we only flag a confident change.
    """
    if baseline_script == "NONE":
        return False
    here = dominant_script(text)
    if here == "NONE":
        return False
    return here != baseline_script
