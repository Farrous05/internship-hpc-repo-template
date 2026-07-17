"""Recovery evaluation after a post-collapse injection.

Vendored from babel-ai (``src/babel_ai/recovery.py``). Given a run that
collapsed, was injected once, and continued, decide whether the model *recovered*
-- moved away from the attractor and stayed diverse, without parroting the
injection or descending into gibberish.

Recovery is declared when, for ``hold_k`` consecutive post-injection rounds, ALL
criteria hold: (1) moved away from the collapsed-window centroid, (2) held for K
rounds, (3) GPT-2 perplexity sane, (4) not parroting the injection, (5) did not
re-collapse, (6) diverse turn-to-turn right now, (7) same language/script.

**Local-embedder swap:** the version-(b) distance needs a *reference embedder*.
babel-ai used OpenAI ``text-embedding-3-large``; the compute nodes are offline,
so this port takes the embedder by **injection** -- call
``set_reference_embedder(model)`` once with any SBERT-compatible model (a local
``SentenceTransformer`` such as BGE/GTE/E5) whose ``.encode(list[str],
convert_to_tensor=True)`` returns a tensor. The ``distance_cutoff`` (0.70) was
calibrated on OpenAI embeddings and MUST be re-tuned for the local model before
recovery numbers are trusted (see analysis/calibrate_reference_embedder in
babel-ai).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from detector.language import dominant_script, switched_language

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+")

# Injected reference embedder (SBERT-compatible: .encode(texts, convert_to_tensor
# =True) -> tensor). Set once via set_reference_embedder() before evaluating
# recovery. Left None so importing this module needs no model / network.
_REFERENCE_EMBEDDER = None


def set_reference_embedder(embedder) -> None:
    """Register the local reference embedder used for version-(b) distances."""
    global _REFERENCE_EMBEDDER
    _REFERENCE_EMBEDDER = embedder


@dataclass(frozen=True)
class RecoveryConfig:
    """Provisional recovery thresholds (calibrate against labeled runs)."""

    hold_k: int = 5  # consecutive qualifying rounds to declare recovery
    distance_cutoff: float = 0.70  # cosine dist from collapsed window (RE-TUNE
    #                                for the local embedder; 0.70 = OpenAI value)
    max_perplexity: float = 150.0  # GPT-2 perplexity gibberish guard
    min_jaccard_to_injection: float = 0.5  # turn must differ from the injection
    min_window_distance: float = 0.40  # "diverse now" == collapse cutoff


@dataclass
class RecoveryResult:
    """Outcome of recovery evaluation for one run."""

    recovered: bool
    recovery_round: Optional[int]  # absolute self-loop round (start of streak)
    hold_length: int  # length of the qualifying streak
    post_injection_distances: List[float] = field(default_factory=list)


def _word_set(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard_distance(a: str, b: str) -> float:
    """1 - |A∩B| / |A∪B| over word sets. 1.0 if either side is empty."""
    sa, sb = _word_set(a), _word_set(b)
    if not sa or not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return 1.0 - (inter / union if union else 0.0)


def _embed(texts: Sequence[str]):
    """Embed texts for the version-(b) distance using the injected local
    reference embedder. Raises if none was registered."""
    if _REFERENCE_EMBEDDER is None:
        raise RuntimeError(
            "No reference embedder set. Call detector.recovery."
            "set_reference_embedder(model) with a local SBERT-compatible model "
            "before evaluating recovery."
        )
    return _REFERENCE_EMBEDDER.encode(list(texts), convert_to_tensor=True)


def version_b_distances(
    contents: Sequence[str], collapsed_window: Sequence[str]
) -> List[float]:
    """Version-(b) distance of each content from the collapsed window centroid.

    ``1 - cosine_similarity`` (clamped) of every entry in ``contents`` to the
    centroid of the (non-empty) collapsed-window turns. ``collapsed_window`` must
    contain at least one non-empty string; callers guard this.
    """

    from sentence_transformers.util import cos_sim

    window = [c for c in collapsed_window if c.strip()]
    centroid = _embed(window).mean(dim=0)
    emb = _embed(list(contents))
    distances: List[float] = []
    for i in range(len(contents)):
        sim = float(cos_sim(emb[i], centroid).item())
        distances.append(1.0 - max(-1.0, min(1.0, sim)))
    return distances


def evaluate_recovery(
    agent_contents: Sequence[str],
    analyses: Sequence[object],
    injection_round: int,
    injection_text: str,
    window: int,
    recollapse_rounds: Optional[Sequence[int]] = None,
    config: Optional[RecoveryConfig] = None,
) -> RecoveryResult:
    """Evaluate whether the run recovered after the injection.

    Args:
        agent_contents: per-round self-loop turn texts (index = round), original
            model outputs (injected span excluded).
        analyses: per-round objects aligned with ``agent_contents`` (perplexity +
            windowed semantic similarity read off them).
        injection_round: round at which injection was applied.
        injection_text: the injected text (criterion 4).
        window: collapsed-window size (turns up to & incl. injection_round).
        recollapse_rounds: rounds at which the online detector declared a NEW
            collapse after injection (criterion 5). Defaults to none.
        config: thresholds; defaults to ``RecoveryConfig()``.
    """

    config = config or RecoveryConfig()
    recollapsed = set(recollapse_rounds or [])
    n = len(agent_contents)

    post_start = injection_round + 1
    if post_start >= n:
        return RecoveryResult(False, None, 0, [])

    win_lo = max(0, injection_round - window + 1)
    collapsed_window = [
        c for c in agent_contents[win_lo : injection_round + 1] if c.strip()
    ]
    post_contents = list(agent_contents[post_start:])
    if not collapsed_window or not any(c.strip() for c in post_contents):
        return RecoveryResult(False, None, 0, [])

    distances = version_b_distances(post_contents, collapsed_window)
    baseline_script = dominant_script(" ".join(collapsed_window))

    streak = 0
    best_start: Optional[int] = None
    best_len = 0
    cur_start = 0
    for i, content in enumerate(post_contents):
        round_idx = post_start + i
        analysis = analyses[round_idx] if round_idx < len(analyses) else None
        perplexity = getattr(analysis, "token_perplexity", None)

        moved_away = distances[i] >= config.distance_cutoff
        perplexity_ok = (
            perplexity is not None and perplexity <= config.max_perplexity
        )
        not_parroting = (
            _jaccard_distance(content, injection_text)
            >= config.min_jaccard_to_injection
        )
        not_recollapsed = round_idx not in recollapsed
        window_sim = getattr(analysis, "semantic_similarity_window", None)
        currently_diverse = (
            window_sim is not None
            and (1.0 - window_sim) >= config.min_window_distance
        )
        same_language = not switched_language(content, baseline_script)

        if (
            moved_away
            and perplexity_ok
            and not_parroting
            and same_language
            and not_recollapsed
            and currently_diverse
        ):
            if streak == 0:
                cur_start = round_idx
            streak += 1
            if streak > best_len:
                best_len = streak
                best_start = cur_start
        else:
            streak = 0

    recovered = best_len >= config.hold_k
    if recovered:
        logger.info(
            "Recovery declared: round %d, held %d rounds", best_start, best_len
        )
    else:
        logger.info("No recovery (longest qualifying streak %d)", best_len)

    return RecoveryResult(
        recovered=recovered,
        recovery_round=best_start if recovered else None,
        hold_length=best_len,
        post_injection_distances=distances,
    )
