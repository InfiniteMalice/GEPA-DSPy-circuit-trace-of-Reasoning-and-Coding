"""Validation helpers for schema V3 objects."""

from __future__ import annotations

import math

from .case_v3 import CASE_NAMES, CaseV3Result
from .rewards import assert_v3_reward_invariants


def validate_case_v3(result: CaseV3Result) -> None:
    """Validate conservative V3 invariants."""

    if result.case_id not in CASE_NAMES:
        raise ValueError(f"unknown base case ID: {result.case_id}")
    assert_v3_reward_invariants(result.reward_components)
    if math.isnan(result.threshold_tau) or result.threshold_tau < 0.0 or result.threshold_tau > 1.0:
        raise ValueError("threshold_tau must be in [0, 1]")


__all__ = ["validate_case_v3"]
