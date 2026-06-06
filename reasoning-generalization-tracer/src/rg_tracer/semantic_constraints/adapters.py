"""Adapters from verified semantic constraints into explicit finite lattices."""

from __future__ import annotations

from collections.abc import Sequence

from rg_tracer.recursive_refinement.lattice import DeductionConstraint, FiniteCandidateLattice

from .compiler import RuleBasedSemanticConstraintCompiler
from .types import SemanticProjectionResult
from .verifier import verify_compilation_result

SEMANTIC_PROJECTION_MODES = frozenset({"off", "shadow", "advisory", "gated_toy_only"})


def _deduction_constraints(result: object) -> list[DeductionConstraint]:
    constraints = []
    for constraint in getattr(result, "constraints", []):
        if not getattr(constraint, "verified", False):
            continue
        constraints.append(
            DeductionConstraint(
                constraint_id=constraint.constraint_id,
                description=(
                    f"Semantic constraint {constraint.relation} from "
                    f"{constraint.provenance.compiler_name}."
                ),
                allowed_candidates=frozenset(constraint.allowed_candidates),
                source="semantic_constraint_synthesis",
            )
        )
    return constraints


def compile_verified_constraints_to_lattice(
    text: str,
    domain: Sequence[object],
    *,
    mode: str = "shadow",
    task_type: str = "semantic_constraint_toy",
) -> SemanticProjectionResult:
    """Compile, verify, and project bounded semantic constraints into a finite lattice."""

    if mode not in SEMANTIC_PROJECTION_MODES:
        raise ValueError(f"mode must be one of {sorted(SEMANTIC_PROJECTION_MODES)}")
    compiler = RuleBasedSemanticConstraintCompiler()
    compilation = compiler.compile(text, domain, task_type=task_type)
    compilation = verify_compilation_result(compilation, domain)
    if mode == "off":
        return SemanticProjectionResult(
            compilation=compilation,
            mode=mode,
            remaining_candidates=list(domain),
            recommended_action="no_compilation",
            lattice_constraints=[],
            gated=False,
        )
    constraints = _deduction_constraints(compilation)
    lattice = FiniteCandidateLattice()
    current = lattice.element(list(domain))
    for constraint in constraints:
        current = lattice.project(current, constraint)
    gated = mode == "gated_toy_only" and compilation.safe_for_gated_projection
    if mode == "gated_toy_only" and not gated:
        action = "refuse_gating"
    elif compilation.contradiction_detected:
        action = "clarification_required"
    elif len(current.candidates) == 1:
        action = "recommend_unique_candidate"
    elif current.candidates:
        action = "recommend_remaining_candidates"
    else:
        action = "clarification_required"
    return SemanticProjectionResult(
        compilation=compilation,
        mode=mode,
        remaining_candidates=sorted(current.candidates, key=repr),
        recommended_action=action,
        lattice_constraints=[constraint.as_dict() for constraint in constraints],
        gated=gated,
    )


__all__ = ["SEMANTIC_PROJECTION_MODES", "compile_verified_constraints_to_lattice"]
