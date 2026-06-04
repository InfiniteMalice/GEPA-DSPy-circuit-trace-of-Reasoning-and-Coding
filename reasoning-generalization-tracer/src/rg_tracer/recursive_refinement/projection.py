"""LDT-inspired explicit projection over task-local finite lattices."""

from __future__ import annotations

from collections.abc import Mapping

from .config import LatticeConfig
from .lattice import (
    FiniteCandidateLattice,
    LatticeProjectionResult,
    ProjectionStep,
    _stable_items,
)
from .lattice_adapters import select_lattice_adapter


class LatticeDeductionProjector:
    """Explicit task-local LDT-inspired projection for bounded toy domains."""

    def __init__(self, config: LatticeConfig | None = None) -> None:
        self.config = config or LatticeConfig()

    def project(self, problem: Mapping[str, object]) -> LatticeProjectionResult:
        config = self.config
        if config.mode == "off":
            return self._empty_result(mode="off")
        adapter = select_lattice_adapter(problem, config.adapter)
        if adapter is None:
            abstain = config.mode == "gated"
            return LatticeProjectionResult(
                initial_candidates=[],
                remaining_candidates=[],
                projection_steps=[],
                meet_count=0,
                join_count=0,
                contradiction_detected=False,
                resolved=False,
                unresolved=True,
                abstain_recommended=abstain,
                abstain_reason="no_reliable_adapter" if abstain else None,
                adapter_name=None,
                mode=config.mode,
            )
        lattice = FiniteCandidateLattice()
        initial = lattice.element(adapter.initial_candidates(problem))
        current = initial
        steps: list[ProjectionStep] = []
        meet_count = 0
        canonicalization_count = 1
        constraints = adapter.constraints(problem)
        for constraint in constraints[: config.max_projection_steps]:
            before = current
            projected = lattice.project(current, constraint)
            meet_count += 1
            canonicalization_count += 1
            removed = before.candidates - projected.candidates
            step = ProjectionStep(
                step_index=len(steps),
                constraint_id=constraint.constraint_id,
                before_candidates=_stable_items(before.candidates),
                after_candidates=_stable_items(projected.candidates),
                removed_candidates=_stable_items(removed),
                operation="meet",
                contradiction=not projected.candidates,
                resolved=len(projected.candidates) == 1,
            )
            steps.append(step)
            current = projected
            if step.contradiction:
                break
        contradiction = not current.candidates
        resolved = len(current.candidates) == 1
        unresolved = bool(current.candidates) and not resolved
        exhausted = len(constraints) > len(steps) and not (contradiction or resolved)
        abstain_reason = None
        if contradiction and config.abstain_on_contradiction:
            abstain_reason = "contradiction"
        elif (unresolved or exhausted) and config.abstain_on_unresolved:
            abstain_reason = "unresolved"
        return LatticeProjectionResult(
            initial_candidates=_stable_items(initial.candidates),
            remaining_candidates=_stable_items(current.candidates),
            projection_steps=steps,
            meet_count=meet_count,
            join_count=0,
            contradiction_detected=contradiction,
            resolved=resolved,
            unresolved=unresolved or exhausted,
            abstain_recommended=abstain_reason is not None,
            abstain_reason=abstain_reason,
            adapter_name=adapter.name,
            mode=config.mode,
            canonicalization_count=canonicalization_count,
        )

    def _empty_result(self, *, mode: str) -> LatticeProjectionResult:
        return LatticeProjectionResult(
            initial_candidates=[],
            remaining_candidates=[],
            projection_steps=[],
            meet_count=0,
            join_count=0,
            contradiction_detected=False,
            resolved=False,
            unresolved=False,
            abstain_recommended=False,
            abstain_reason=None,
            mode=mode,
        )


__all__ = ["LatticeDeductionProjector"]
