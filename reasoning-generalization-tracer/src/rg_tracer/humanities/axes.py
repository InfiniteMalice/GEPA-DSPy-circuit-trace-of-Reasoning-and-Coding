"""Deterministic humanities rigor scoring axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

HUMANITIES_AXES = [
    "source_handling",
    "interpretive_fidelity",
    "historiography_context",
    "causal_discipline",
    "triangulation",
    "normative_positive_sep",
    "uncertainty_calibration",
    "intellectual_charity",
    "rhetorical_hygiene",
    "reproducibility_transparency",
    "synthesis_generalization",
    "epistemic_neutrality",
]


def _coerce_score(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 <= value <= 4:
        return value
    return None


@dataclass(frozen=True)
class HumanitiesScores:
    scores: Mapping[str, int]

    def passes_hard_gates(self) -> bool:
        return (
            self.scores.get("source_handling", 0) >= 3
            and self.scores.get("interpretive_fidelity", 0) >= 3
            and self.scores.get("normative_positive_sep", 0) >= 2
            and self.scores.get("uncertainty_calibration", 0) >= 2
        )


def _axis_score(metrics: Mapping[str, object], *, penalties: int = 0) -> int:
    base = _coerce_score(metrics.get("score"))
    if base is not None:
        return base
    positive = float(metrics.get("positive", 0.0) or 0.0)
    coverage = float(metrics.get("coverage", 0.0) or 0.0)
    deductions = int(metrics.get("deductions", penalties) or penalties)
    score = 0
    if positive >= 0.9 and coverage >= 0.8 and deductions == 0:
        score = 4
    elif positive >= 0.75 and coverage >= 0.7:
        score = 3
    elif positive >= 0.6:
        score = 2
    elif positive >= 0.4:
        score = 1
    if deductions:
        score = max(0, score - min(deductions, 3))
    return score


def score_axis(name: str, metrics: Mapping[str, object]) -> int:
    penalties = int(metrics.get("penalties", 0) or 0)
    return _axis_score(metrics, penalties=penalties)


__all__ = ["HUMANITIES_AXES", "HumanitiesScores", "score_axis"]
