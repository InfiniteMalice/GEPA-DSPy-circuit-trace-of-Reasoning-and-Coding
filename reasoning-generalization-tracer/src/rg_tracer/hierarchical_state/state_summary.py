"""Folded state records for compact long-horizon context."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass
class FoldedState:
    task_id: str
    completed_summary: str
    active_subgoals: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_constraints: List[str] = field(default_factory=list)
    failed_attempts: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "completed_summary": self.completed_summary,
            "active_subgoals": list(self.active_subgoals),
            "unresolved_constraints": list(self.unresolved_constraints),
            "failed_attempts": list(self.failed_attempts),
            "evidence_refs": list(self.evidence_refs),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FoldedState":
        return cls(
            task_id=str(data["task_id"]),
            completed_summary=str(data.get("completed_summary", "")),
            active_subgoals=list(data.get("active_subgoals", []) or []),
            unresolved_constraints=list(data.get("unresolved_constraints", []) or []),
            failed_attempts=list(data.get("failed_attempts", []) or []),
            evidence_refs=list(data.get("evidence_refs", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "FoldedState":
        return cls.from_mapping(json.loads(text))


__all__ = ["FoldedState"]
