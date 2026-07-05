"""Shared perturbation record helpers."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


@dataclass
class PerturbedTask:
    task_id: str
    parent_task_id: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "prompt": self.prompt,
            "metadata": dict(self.metadata),
        }


def make_task(
    task: Mapping[str, Any],
    shift_type: str,
    prompt: str,
    seed: int = 0,
    **metadata: Any,
) -> PerturbedTask:
    parent_id = str(task.get("task_id", "unknown"))
    suffix = uuid.uuid4().hex[:12]
    return PerturbedTask(
        task_id=f"{parent_id}:{shift_type}:{suffix}",
        parent_task_id=parent_id,
        prompt=prompt,
        metadata={"shift_type": shift_type, "seed": seed, **metadata},
    )


__all__ = ["PerturbedTask", "make_task"]
