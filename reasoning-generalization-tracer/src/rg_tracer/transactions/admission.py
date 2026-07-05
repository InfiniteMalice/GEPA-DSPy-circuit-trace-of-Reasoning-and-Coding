"""Admission control for generated proposals."""

from __future__ import annotations

from typing import Dict, Iterable

from .constraints import ConstraintCheck, default_constraints
from .proposal import AdmissionDecision, Proposal


def admit_proposal(
    proposal: Proposal,
    constraints: Iterable[ConstraintCheck] | None = None,
    config: Dict[str, object] | None = None,
) -> AdmissionDecision:
    """Run deterministic checks before accepting a proposal as effective state."""

    active = list(constraints or default_constraints())
    settings = dict(config or {})
    violations: list[str] = []
    for constraint in active:
        passed, reason = constraint.run(proposal, settings)
        if not passed:
            violations.append(f"{constraint.name}: {reason}")
    accepted = not violations
    reason = "accepted" if accepted else "rejected by deterministic admission constraints"
    warnings = list(proposal.metadata.get("warnings", []) or [])
    return AdmissionDecision(
        accepted=accepted,
        reason=reason,
        violated_constraints=violations,
        warnings=warnings,
        metadata={"constraint_count": len(active)},
    )


__all__ = ["admit_proposal"]
