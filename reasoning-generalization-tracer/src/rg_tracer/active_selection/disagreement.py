"""Disagreement scoring for candidate predictions."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


def disagreement_score(predictions: Iterable[str]) -> float:
    items = list(predictions)
    if not items:
        return 0.0
    counts = Counter(items)
    majority = counts.most_common(1)[0][1]
    return 1.0 - majority / len(items)


__all__ = ["disagreement_score"]
