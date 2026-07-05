"""Deterministic proposal constraint checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from .proposal import Proposal

ConstraintFn = Callable[[Proposal, Dict[str, object]], tuple[bool, str | None]]


@dataclass
class ConstraintCheck:
    name: str
    description: str
    check: ConstraintFn

    def run(self, proposal: Proposal, config: Dict[str, object]) -> tuple[bool, str | None]:
        return self.check(proposal, config)


def default_constraints() -> List[ConstraintCheck]:
    return [
        ConstraintCheck("required_fields", "Required proposal fields are present", _required),
        ConstraintCheck("verifier_passed", "Verifier diagnostics passed", _verifier_passed),
        ConstraintCheck("score_threshold", "Score is above configured threshold", _score_ok),
        ConstraintCheck(
            "contradiction_limit", "Contradiction warnings are bounded", _contradiction_ok
        ),
        ConstraintCheck("claim_drift_limit", "Claim drift warnings are bounded", _claim_drift_ok),
        ConstraintCheck(
            "provenance", "Accepted skill/reference updates keep provenance", _provenance
        ),
    ]


def _required(proposal: Proposal, _config: Dict[str, object]) -> tuple[bool, str | None]:
    if proposal.proposal_id and proposal.task_id and proposal.proposal_type:
        return True, None
    return False, "proposal requires id, task_id, and proposal_type"


def _verifier_passed(proposal: Proposal, config: Dict[str, object]) -> tuple[bool, str | None]:
    if not config.get("require_verifier", False):
        return True, None
    if proposal.metadata.get("verifier_passed") is True:
        return True, None
    return False, "verifier did not pass"


def _score_ok(proposal: Proposal, config: Dict[str, object]) -> tuple[bool, str | None]:
    threshold = float(config.get("score_threshold", 0.0))
    score = float(proposal.metadata.get("score", 1.0))
    if score >= threshold:
        return True, None
    return False, f"score {score} below threshold {threshold}"


def _contradiction_ok(proposal: Proposal, config: Dict[str, object]) -> tuple[bool, str | None]:
    max_severity = str(config.get("max_contradiction_severity", "high"))
    severity = str(proposal.metadata.get("contradiction_severity", "none"))
    return _severity_allowed(severity, max_severity, "contradiction severity too high")


def _claim_drift_ok(proposal: Proposal, config: Dict[str, object]) -> tuple[bool, str | None]:
    max_severity = str(config.get("max_claim_drift_severity", "high"))
    severity = str(proposal.metadata.get("claim_drift_severity", "none"))
    return _severity_allowed(severity, max_severity, "claim drift severity too high")


def _provenance(proposal: Proposal, _config: Dict[str, object]) -> tuple[bool, str | None]:
    if proposal.proposal_type not in {"skill_update", "reference_update"}:
        return True, None
    if proposal.metadata.get("provenance") or proposal.payload.get("provenance"):
        return True, None
    return False, "accepted skill/reference updates require provenance"


def _severity_allowed(
    severity: str,
    max_severity: str,
    message: str,
) -> tuple[bool, str | None]:
    order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    if severity not in order:
        return False, f"unknown severity: {severity}"
    if max_severity not in order:
        return False, f"unknown max severity: {max_severity}"
    if order[severity] <= order[max_severity]:
        return True, None
    return False, message


__all__ = ["ConstraintCheck", "default_constraints"]
