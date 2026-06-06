"""Verifier for bounded semantic constraints."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from .types import SemanticCompilationResult, SemanticConstraint

ALLOWED_RELATIONS = frozenset(
    {
        "is_even",
        "is_odd",
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
        "not_equal",
        "one_of",
    }
)


def _hashable_domain(domain: Sequence[object]) -> set[object]:
    values = set()
    for item in domain:
        try:
            hash(item)
        except TypeError:
            continue
        values.add(item)
    return values


def _verify_constraint(
    constraint: SemanticConstraint,
    domain_values: set[object],
) -> tuple[SemanticConstraint, list[str]]:
    errors = []
    updated = deepcopy(constraint)
    allowed = set(updated.allowed_candidates)
    if updated.relation not in ALLOWED_RELATIONS:
        errors.append("unsupported_relation")
    if updated.subject != "answer":
        errors.append("unsupported_subject")
    if not allowed:
        errors.append("empty_candidate_set")
    if not allowed <= domain_values:
        errors.append("candidate_outside_domain")
    if not 0.0 <= updated.confidence <= 1.0:
        errors.append("confidence_out_of_bounds")
    if not updated.provenance.source_text:
        errors.append("missing_provenance")
    updated.verified = not errors
    updated.verification_status = "verified" if updated.verified else ",".join(errors)
    return updated, errors


def verify_compilation_result(
    result: SemanticCompilationResult,
    domain: Sequence[object],
) -> SemanticCompilationResult:
    """Verify constraints against grammar, provenance, duplicates, and contradictions."""

    domain_values = _hashable_domain(domain)
    verified_constraints = []
    errors: list[str] = []
    seen: set[tuple[str, str, tuple[object, ...]]] = set()
    duplicates = []
    remaining = set(domain_values)
    for constraint in result.constraints:
        updated, constraint_errors = _verify_constraint(constraint, domain_values)
        key = (
            updated.relation,
            repr(updated.object_value),
            tuple(sorted(updated.allowed_candidates, key=repr)),
        )
        if key in seen:
            updated.verified = False
            updated.verification_status = "duplicate_constraint"
            duplicates.append(updated.constraint_id)
        seen.add(key)
        if constraint_errors:
            errors.extend(constraint_errors)
        if updated.verified:
            remaining &= set(updated.allowed_candidates)
        verified_constraints.append(updated)
    contradiction = bool(verified_constraints) and not remaining
    verification_errors = sorted(set(errors))
    if duplicates:
        verification_errors.append("duplicate_constraint")
    if contradiction:
        verification_errors.append("contradiction")
    safe_for_shadow = bool(verified_constraints) and not result.ambiguity_detected
    safe_for_gated = (
        safe_for_shadow
        and not contradiction
        and not result.unsupported_fragments
        and all(constraint.verified for constraint in verified_constraints)
        and result.task_type == "semantic_constraint_toy"
    )
    status = "verified" if safe_for_gated else "requires_review"
    if not verified_constraints:
        status = "unverified"
    verified_count = sum(1 for constraint in verified_constraints if constraint.verified)
    result.constraints = verified_constraints
    result.contradiction_detected = contradiction
    result.verification_status = status
    result.safe_for_shadow_projection = safe_for_shadow
    result.safe_for_gated_projection = safe_for_gated
    result.diagnostics = {
        **result.diagnostics,
        "verification_errors": verification_errors,
        "duplicate_constraint_ids": duplicates,
        "verified_constraint_count": verified_count,
        "remaining_candidate_count": len(remaining) if not contradiction else 0,
    }
    return result


__all__ = ["ALLOWED_RELATIONS", "verify_compilation_result"]
