"""Reward invariants for schema V3."""

from __future__ import annotations

from .case_v3 import RewardComponents


def assert_v3_reward_invariants(components: RewardComponents) -> None:
    """Validate V3 reward invariants that protect hidden thought traces."""

    if components.r_thought < 0.0:
        raise ValueError("r_thought must never be negative")


__all__ = ["assert_v3_reward_invariants"]
