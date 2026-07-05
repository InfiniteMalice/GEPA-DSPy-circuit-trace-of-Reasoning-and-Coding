"""Deterministic claim-drift reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ClaimDriftReport:
    task_id: str
    claim: str
    artifact_ref: str
    drift_type: str
    severity: str
    evidence: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "claim": self.claim,
            "artifact_ref": self.artifact_ref,
            "drift_type": self.drift_type,
            "severity": self.severity,
            "evidence": list(self.evidence),
            "recommendation": self.recommendation,
        }


def check_claim_drift(
    task_id: str,
    claim: str,
    artifact_ref: str,
    diagnostics: Dict[str, Any],
) -> ClaimDriftReport:
    unsupported = diagnostics.get("unsupported_claims", [])
    if claim in unsupported:
        return ClaimDriftReport(
            task_id=task_id,
            claim=claim,
            artifact_ref=artifact_ref,
            drift_type="unsupported_claim",
            severity="high",
            evidence=[claim],
            recommendation="add evidence or narrow the claim",
        )
    drift_type = str(diagnostics.get("drift_type", "none"))
    severity = "none" if drift_type == "none" else str(diagnostics.get("severity", "medium"))
    return ClaimDriftReport(
        task_id=task_id,
        claim=claim,
        artifact_ref=artifact_ref,
        drift_type=drift_type,
        severity=severity,
        evidence=list(diagnostics.get("evidence", []) or []),
        recommendation=str(diagnostics.get("recommendation", "")),
    )


__all__ = ["ClaimDriftReport", "check_claim_drift"]
