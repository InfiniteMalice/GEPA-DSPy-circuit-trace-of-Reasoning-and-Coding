"""Score helpers for assisted attempts."""

from __future__ import annotations

from .types import AssistedAttemptRecord


def assisted_improvement(record: AssistedAttemptRecord) -> float:
    return record.post_guidance_score - record.pre_guidance_score


def apply_reward_decay(score: float, guidance_used: bool, decay: float) -> float:
    if not guidance_used:
        return score
    bounded = max(0.0, min(1.0, decay))
    return score * (1.0 - bounded)


__all__ = ["apply_reward_decay", "assisted_improvement"]
