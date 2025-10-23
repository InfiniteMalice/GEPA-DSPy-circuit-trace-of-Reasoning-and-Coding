"""Confidence-aware abstention policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

ABSTENTION_THRESHOLD = 0.75


@dataclass
class AbstentionResult:
    text: str
    abstained: bool
    confidence: float


def apply_abstention(
    output_text: str,
    confidence: float,
    sem_score: float,
    gates_pass: bool = True,
) -> AbstentionResult:
    """Return abstention result enforcing the 0.75 threshold and semantic gate."""
    should_abstain = confidence < ABSTENTION_THRESHOLD or sem_score < 2 or not gates_pass
    if should_abstain:
        return AbstentionResult(text="I don't know.", abstained=True, confidence=confidence)
    return AbstentionResult(text=output_text, abstained=False, confidence=confidence)


def apply_abstention_tuple(
    output_text: str,
    confidence: float,
    sem_score: float,
    gates_pass: bool = True,
) -> Tuple[str, bool]:
    result = apply_abstention(output_text, confidence, sem_score, gates_pass)
    return result.text, result.abstained


__all__ = [
    "apply_abstention",
    "apply_abstention_tuple",
    "AbstentionResult",
    "ABSTENTION_THRESHOLD",
]
