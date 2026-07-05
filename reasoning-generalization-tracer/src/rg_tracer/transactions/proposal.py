"""Generic proposal and admission records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Proposal:
    proposal_id: str
    task_id: str
    proposal_type: str
    payload: Dict[str, Any]
    created_at: str = field(default_factory=utc_now)
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "task_id": self.task_id,
            "proposal_type": self.proposal_type,
            "payload": dict(self.payload),
            "created_at": self.created_at,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Proposal":
        return cls(
            proposal_id=str(data["proposal_id"]),
            task_id=str(data["task_id"]),
            proposal_type=str(data["proposal_type"]),
            payload=dict(data.get("payload", {}) or {}),
            created_at=str(data.get("created_at") or utc_now()),
            source=str(data.get("source", "unknown")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class AdmissionDecision:
    accepted: bool
    reason: str
    violated_constraints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "violated_constraints": list(self.violated_constraints),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


__all__ = ["AdmissionDecision", "Proposal", "utc_now"]
