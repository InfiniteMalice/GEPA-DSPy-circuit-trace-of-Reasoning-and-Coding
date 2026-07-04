"""Teacher-dependency metrics for Socratic guidance."""

from __future__ import annotations

from typing import Iterable

from .types import AssistedAttemptRecord


def teacher_dependency_metrics(records: Iterable[AssistedAttemptRecord]) -> dict[str, float]:
    items = list(records)
    if not items:
        return {
            "first_pass_success_rate": 0.0,
            "post_guidance_success_rate": 0.0,
            "guidance_dependency_rate": 0.0,
            "retained_improvement_rate": 0.0,
        }
    first_pass = [
        item for item in items if not item.guidance_used and item.pre_guidance_score >= 1.0
    ]
    post_guidance = [
        item for item in items if item.guidance_used and item.post_guidance_score >= 1.0
    ]
    dependent = [
        item
        for item in items
        if item.guidance_used and item.pre_guidance_score < 1.0 <= item.post_guidance_score
    ]
    retained = [
        item for item in items if not item.guidance_used and item.post_guidance_score >= 1.0
    ]
    total = len(items)
    return {
        "first_pass_success_rate": len(first_pass) / total,
        "post_guidance_success_rate": len(post_guidance) / total,
        "guidance_dependency_rate": len(dependent) / total,
        "retained_improvement_rate": len(retained) / total,
    }


__all__ = ["teacher_dependency_metrics"]
