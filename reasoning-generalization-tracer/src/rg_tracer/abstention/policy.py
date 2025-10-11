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


def apply_abstention(output_text: str, confidence: float) -> AbstentionResult:
    """Return abstention result enforcing the 0.75 confidence threshold."""
    if confidence < ABSTENTION_THRESHOLD:
        return AbstentionResult(text="I don't know.", abstained=True, confidence=confidence)
    return AbstentionResult(text=output_text, abstained=False, confidence=confidence)


__all__ = ["apply_abstention", "AbstentionResult", "ABSTENTION_THRESHOLD"]
