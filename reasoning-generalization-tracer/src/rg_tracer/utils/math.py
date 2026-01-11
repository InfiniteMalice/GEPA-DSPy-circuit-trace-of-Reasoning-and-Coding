"""Numeric helpers shared across scoring modules."""

from __future__ import annotations

import math
from typing import Mapping


def weighted_geometric_mean(
    axis_scores: Mapping[str, float],
    weights: Mapping[str, float],
    *,
    epsilon: float,
) -> float:
    """Compute the weighted geometric mean of ``axis_scores`` with stability checks."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not weights:
        raise ValueError("weights must not be empty")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("total weight must be positive")
    log_sum = 0.0
    for axis, weight in weights.items():
        score = float(axis_scores.get(axis, 0.0))
        shifted = score + epsilon
        if shifted <= 0:
            raise ValueError(f"axis score for {axis!r} must be > {-epsilon}")
        log_sum += (weight / total_weight) * math.log(shifted)
    return math.exp(log_sum)


__all__ = ["weighted_geometric_mean"]
