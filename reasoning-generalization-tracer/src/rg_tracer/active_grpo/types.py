"""Policy-agnostic Active-GRPO reference update records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ReferenceRecord:
    task_id: str
    reference_answer: str
    reference_score: float
    source: str
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    lineage: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "reference_answer": self.reference_answer,
            "reference_score": float(self.reference_score),
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "lineage": list(self.lineage),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ReferenceRecord":
        return cls(
            task_id=str(data["task_id"]),
            reference_answer=str(data["reference_answer"]),
            reference_score=float(data["reference_score"]),
            source=str(data.get("source", "unknown")),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            lineage=list(data.get("lineage", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class CandidateRecord:
    task_id: str
    candidate_answer: str
    candidate_score: float
    verified: bool
    verification_diagnostics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveGRPODecision:
    mode: str
    reason: str
    score_delta: float
    should_update_reference: bool


__all__ = ["ActiveGRPODecision", "CandidateRecord", "ReferenceRecord", "utc_now"]
