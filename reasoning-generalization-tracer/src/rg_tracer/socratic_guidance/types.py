"""Records for assisted attempts and Socratic guidance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class GuidanceRecord:
    task_id: str
    attempt_id: str
    failure_summary: str
    guidance_text: str
    guidance_type: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistedAttemptRecord:
    task_id: str
    attempt_id: str
    pre_guidance_score: float
    post_guidance_score: float
    guidance_used: bool
    reward_decay: float
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["AssistedAttemptRecord", "GuidanceRecord"]
