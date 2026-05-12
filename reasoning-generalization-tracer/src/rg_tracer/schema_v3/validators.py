"""Validation helpers for schema V3 objects."""

from __future__ import annotations

from .case_v3 import CASE_NAMES, CaseV3Result


def validate_case_v3(result: CaseV3Result) -> None:
    """Validate conservative V3 invariants."""

    if result.case_id not in CASE_NAMES:
        raise ValueError(f"unknown base case ID: {result.case_id}")
    if result.reward_components.r_thought < 0.0:
        raise ValueError("negative hidden-thought reward is forbidden")
    if result.threshold_tau < 0.0 or result.threshold_tau > 1.0:
        raise ValueError("threshold_tau must be in [0, 1]")


__all__ = ["validate_case_v3"]
