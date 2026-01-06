"""JSONL logging helpers for DAPO hybrid training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from gepa_dapo_grn.gepa_interfaces import GEPAFeedback


def _feedback_to_dict(feedback: GEPAFeedback) -> Dict[str, Any]:
    """Convert a GEPAFeedback instance to a serializable dictionary."""
    return {
        "rewards": dict(feedback.rewards),
        "tags": dict(feedback.tags),
        "meta": dict(feedback.meta),
        "abstained": bool(feedback.abstained),
    }


def build_log_record(
    *,
    step: int,
    rl_metrics: Mapping[str, Any],
    feedbacks: Sequence[GEPAFeedback],
    generation_metadata: Sequence[Mapping[str, Any]],
    curriculum: Optional[Mapping[str, Any]] = None,
    safety: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a structured log record for DAPO hybrid training."""
    record: Dict[str, Any] = {
        "step": step,
        "rl": dict(rl_metrics),
        "curriculum": dict(curriculum or {}),
        "safety": dict(safety or {}),
        "gepa": [_feedback_to_dict(feedback) for feedback in feedbacks],
        "generation": [dict(meta) for meta in generation_metadata],
    }
    return record


class JSONLLogger:
    """Append-only JSONL logger for DAPO hybrid training metrics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Mapping[str, Any]) -> None:
        def _default(obj: Any) -> Any:
            if hasattr(obj, "item"):
                try:
                    return obj.item()
                except (ValueError, RuntimeError):
                    pass
            return str(obj)

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=_default))
            handle.write("\n")

    def write_step(
        self,
        *,
        step: int,
        rl_metrics: Mapping[str, Any],
        feedbacks: Sequence[GEPAFeedback],
        generation_metadata: Sequence[Mapping[str, Any]],
        curriculum: Optional[Mapping[str, Any]] = None,
        safety: Optional[Mapping[str, Any]] = None,
    ) -> None:
        record = build_log_record(
            step=step,
            rl_metrics=rl_metrics,
            feedbacks=feedbacks,
            generation_metadata=generation_metadata,
            curriculum=curriculum,
            safety=safety,
        )
        self.write(record)


def summarize_task_emas(ema_by_task: Mapping[str, float]) -> Dict[str, float]:
    """Convert task EMAs to a dict with float values (not numpy scalars)."""
    return {task: float(value) for task, value in ema_by_task.items()}
