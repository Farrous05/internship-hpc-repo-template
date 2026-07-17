"""Collapse detection for self-looping LLM experiments.

Vendored verbatim from babel-ai (``src/babel_ai/collapse.py``) — pure numpy, no
ML deps. The **primary** trigger is the windowed cosine distance
``1 - semantic_similarity_window``; the windowed Jaccard distance is tracked as a
secondary corroborating signal but does not gate the decision. Fed one round at a
time so it can declare collapse *online*. Round indices count from the first
self-loop turn (0 = first agent-generated message).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollapseConfig:
    """Calibrated detector constants (single source of truth, Part A).

    Distances are ``1 - similarity``; lower = more collapsed.
    """

    window: int = 10  # W: sliding window for the windowed metrics
    persistence: int = 3  # K: consecutive rounds below cutoff to declare
    cosine_cutoff: float = 0.40  # windowed cosine DISTANCE threshold (primary)
    jaccard_cutoff: float = 0.40  # windowed Jaccard DISTANCE (secondary/log)
    warmup: int = 10  # no collapse declared before this round (window fills)
    rearm_cutoff: float = 0.50  # hysteresis re-arm band (must exceed cutoff)


@dataclass
class CollapseDetector:
    """Online collapse detector.

    Usage::

        det = CollapseDetector()
        for round_idx, analysis in enumerate(per_round_analyses):
            det.update_from_analysis(round_idx, analysis)
        det.onset_round      # TTC, or None
        det.collapse_rate()  # slope of semantic similarity to onset
    """

    config: CollapseConfig = field(default_factory=CollapseConfig)

    def __post_init__(self) -> None:
        self._cos_streak = 0
        self._jac_streak = 0
        self.onset_round: Optional[int] = None
        self.jaccard_onset_round: Optional[int] = None
        self._armed: bool = True
        self.last_rearm_round: Optional[int] = None
        self._sem_history: List[Tuple[int, float]] = []

    @property
    def collapsed(self) -> bool:
        """Whether collapse has been declared (primary signal)."""
        return self.onset_round is not None

    def rearm(self, round_idx: Optional[int] = None) -> None:
        """Reset collapse state so a *new* collapse can be detected, under
        hysteresis (used after a repeated post-collapse injection)."""
        self._cos_streak = 0
        self._jac_streak = 0
        self.onset_round = None
        self.jaccard_onset_round = None
        self._armed = False
        self.last_rearm_round = round_idx

    def update(
        self,
        round_idx: int,
        semantic_similarity_window: Optional[float],
        semantic_similarity: Optional[float] = None,
        lexical_similarity_window: Optional[float] = None,
    ) -> bool:
        """Feed one round's metrics; return whether collapse is declared so far.

        ``round_idx`` counts from 0 at the first self-loop turn. Only the
        primary (windowed cosine) signal sets ``onset_round``; the Jaccard
        signal is tracked separately for corroboration.
        """

        if semantic_similarity is not None:
            self._sem_history.append((round_idx, float(semantic_similarity)))

        # Hysteresis gate: while disarmed (just after an injection), watch for
        # the windowed cosine distance to climb above rearm_cutoff before a new
        # collapse can be looked for.
        if not self._armed and semantic_similarity_window is not None:
            if 1.0 - float(semantic_similarity_window) > self.config.rearm_cutoff:
                self._armed = True
                self._cos_streak = 0
                self._jac_streak = 0
                logger.info(
                    "Detector re-armed at round %d (windowed cosine distance "
                    "rose above %.2f -- loop left the attractor)",
                    round_idx,
                    self.config.rearm_cutoff,
                )

        # Primary trigger: windowed cosine distance. Only counts while armed.
        if self._armed and semantic_similarity_window is not None:
            cos_dist = 1.0 - float(semantic_similarity_window)
            if cos_dist <= self.config.cosine_cutoff:
                self._cos_streak += 1
                if (
                    self._cos_streak >= self.config.persistence
                    and round_idx >= self.config.warmup
                    and self.onset_round is None
                ):
                    self.onset_round = round_idx
                    logger.info(
                        "Collapse declared at round %d (windowed cosine "
                        "distance %.3f <= %.2f for %d consecutive rounds)",
                        round_idx,
                        cos_dist,
                        self.config.cosine_cutoff,
                        self.config.persistence,
                    )
            else:
                self._cos_streak = 0

        # Secondary signal: windowed Jaccard distance (logged, never a trigger).
        if self._armed and lexical_similarity_window is not None:
            jac_dist = 1.0 - float(lexical_similarity_window)
            if jac_dist <= self.config.jaccard_cutoff:
                self._jac_streak += 1
                if (
                    self._jac_streak >= self.config.persistence
                    and round_idx >= self.config.warmup
                    and self.jaccard_onset_round is None
                ):
                    self.jaccard_onset_round = round_idx
            else:
                self._jac_streak = 0

        return self.collapsed

    def update_from_analysis(self, round_idx: int, analysis: object) -> bool:
        """Convenience wrapper that pulls fields off an ``AnalysisResult``."""

        return self.update(
            round_idx=round_idx,
            semantic_similarity_window=getattr(
                analysis, "semantic_similarity_window", None
            ),
            semantic_similarity=getattr(analysis, "semantic_similarity", None),
            lexical_similarity_window=getattr(
                analysis, "lexical_similarity_window", None
            ),
        )

    def collapse_rate(self) -> Optional[float]:
        """Collapse rate = slope of turn-to-turn semantic similarity from
        round 0 to onset (TTC). ``None`` if collapse never declared or too few
        points."""

        if self.onset_round is None:
            return None
        pts = [(r, s) for r, s in self._sem_history if r <= self.onset_round]
        if len(pts) < 2:
            return None
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)
        return float(np.polyfit(x, y, 1)[0])
