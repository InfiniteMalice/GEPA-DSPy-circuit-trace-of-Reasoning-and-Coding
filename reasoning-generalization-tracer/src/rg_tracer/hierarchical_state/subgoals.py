"""Subgoal records for long-horizon task state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Subgoal:
    subgoal_id: str
    task_id: str
    description: str
    status: str
    evidence_refs: List[str] = field(default_factory=list)
    open_constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgoal_id": self.subgoal_id,
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status,
            "evidence_refs": list(self.evidence_refs),
            "open_constraints": list(self.open_constraints),
            "metadata": dict(self.metadata),
        }


__all__ = ["Subgoal"]
