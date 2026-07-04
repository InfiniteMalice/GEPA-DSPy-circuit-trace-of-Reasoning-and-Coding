"""Repair trace serialization for validation harnesses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass
class RepairTrace:
    task_id: str
    attempt_id: str
    failure_summary: str
    repair_summary: str
    tests_run: List[str]
    result: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "failure_summary": self.failure_summary,
            "repair_summary": self.repair_summary,
            "tests_run": list(self.tests_run),
            "result": self.result,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RepairTrace":
        return cls(
            task_id=str(data["task_id"]),
            attempt_id=str(data["attempt_id"]),
            failure_summary=str(data.get("failure_summary", "")),
            repair_summary=str(data.get("repair_summary", "")),
            tests_run=list(data.get("tests_run", []) or []),
            result=str(data.get("result", "")),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "RepairTrace":
        return cls.from_mapping(json.loads(text))


__all__ = ["RepairTrace"]
