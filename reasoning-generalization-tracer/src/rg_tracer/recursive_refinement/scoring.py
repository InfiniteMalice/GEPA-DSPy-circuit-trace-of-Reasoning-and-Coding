"""Heuristic process scoring for recursive-refinement trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .types import RefinementState


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class ProcessScore:
    """Decomposed public process score, not a learned reward model."""

    total: float
    confidence: float
    uncertainty_reduction: float
    verification: float
    convergence: float
    efficiency: float
    diversity: float
    redundancy: float
    diagnostics: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_trajectory(
    states: list[RefinementState],
    *,
    total_updates: int,
    max_total_updates: int,
    converged: bool,
    budget_exhausted: bool,
) -> ProcessScore:
    """Score a trajectory using explicit inspectable process factors."""

    if not states:
        return ProcessScore(
            total=0.0,
            confidence=0.0,
            uncertainty_reduction=0.0,
            verification=0.0,
            convergence=0.0,
            efficiency=0.0,
            diversity=0.0,
            redundancy=0.0,
            diagnostics={"empty": True},
        )
    first = states[0]
    final = states[-1]
    operations = final.operations
    view_names = [operation.view_name for operation in operations]
    verification_hits = [
        operation.constraint_passed
        for operation in operations
        if operation.view_name in {"verification", "constraint_check"}
    ]
    contradictions = sum(1 for operation in operations if operation.constraint_passed is False)
    repeated = max(0, len(view_names) - len(set(view_names)))
    confidence = _bounded(final.confidence)
    uncertainty_reduction = _bounded(first.uncertainty - final.uncertainty)
    verification = 1.0 if verification_hits and all(verification_hits) else 0.0
    if not verification_hits:
        verification = 0.5
    convergence = 1.0 if converged else 0.0
    efficiency = 1.0 - min(1.0, total_updates / max(max_total_updates, 1))
    diversity = _bounded(len(set(view_names)) / 4.0)
    redundancy = -min(1.0, repeated / max(len(view_names), 1))
    budget_penalty = -0.25 if budget_exhausted and uncertainty_reduction <= 0.0 else 0.0
    contradiction_penalty = -0.15 * contradictions
    total = (
        0.30 * confidence
        + 0.20 * uncertainty_reduction
        + 0.20 * verification
        + 0.10 * convergence
        + 0.10 * efficiency
        + 0.10 * diversity
        + 0.05 * redundancy
        + budget_penalty
        + contradiction_penalty
    )
    return ProcessScore(
        total=float(total),
        confidence=confidence,
        uncertainty_reduction=uncertainty_reduction,
        verification=verification,
        convergence=convergence,
        efficiency=efficiency,
        diversity=diversity,
        redundancy=redundancy,
        diagnostics={
            "contradiction_count": contradictions,
            "repeated_operation_count": repeated,
            "budget_exhausted_without_progress": budget_penalty < 0.0,
            "verification_checks": len(verification_hits),
        },
    )


__all__ = ["ProcessScore", "score_trajectory"]
