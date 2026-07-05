"""Socratic guidance metadata and dependency metrics."""

from .dependency_metrics import teacher_dependency_metrics
from .guidance import apply_reward_decay, assisted_improvement
from .types import AssistedAttemptRecord, GuidanceRecord

__all__ = [
    "AssistedAttemptRecord",
    "GuidanceRecord",
    "apply_reward_decay",
    "assisted_improvement",
    "teacher_dependency_metrics",
]
