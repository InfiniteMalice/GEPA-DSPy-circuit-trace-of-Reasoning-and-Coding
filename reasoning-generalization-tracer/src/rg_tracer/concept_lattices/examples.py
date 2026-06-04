"""Synthetic conceptual lattice examples for shadow diagnostics only."""

from __future__ import annotations

from .types import ConceptAttribute, ConceptLatticeExample, ConceptLatticeSpec


def synthetic_examples() -> tuple[ConceptLatticeSpec, ...]:
    """Return explicit human-authored examples that do not gate outputs."""

    return (
        ConceptLatticeSpec(
            name="code_correctness",
            domain="code correctness",
            attributes=(
                ConceptAttribute("precondition_checked", "Inputs are validated before use."),
                ConceptAttribute("postcondition_verified", "Output constraints are checked."),
                ConceptAttribute("edge_case_considered", "Boundary cases are represented."),
            ),
            implications=(("postcondition_verified", "edge_case_considered"),),
        ),
        ConceptLatticeSpec(
            name="security_permissions",
            domain="security permissions",
            attributes=(
                ConceptAttribute("least_privilege", "The action uses minimal permissions."),
                ConceptAttribute("authorization_checked", "Authority is explicit and current."),
                ConceptAttribute("audit_visible", "The decision is externally inspectable."),
            ),
        ),
        ConceptLatticeSpec(
            name="causal_reasoning",
            domain="causal reasoning",
            attributes=(
                ConceptAttribute("confounder_considered", "A confounder is represented."),
                ConceptAttribute("intervention_named", "An intervention is explicit."),
                ConceptAttribute("counterfactual_checked", "A counterfactual is evaluated."),
            ),
        ),
        ConceptLatticeSpec(
            name="ambiguity_authority",
            domain="ambiguity and authority checks",
            attributes=(
                ConceptAttribute("ambiguity_detected", "Material ambiguity is identified."),
                ConceptAttribute("authority_scoped", "Authority and scope are bounded."),
                ConceptAttribute("clarification_available", "A clarifying route is available."),
            ),
        ),
    )


EXAMPLE_CASES = (
    ConceptLatticeExample(
        example_id="code_edge_case",
        prompt="A function divides user input after checking for zero.",
        expected_attributes=("precondition_checked", "edge_case_considered"),
        lattice_name="code_correctness",
    ),
    ConceptLatticeExample(
        example_id="permission_scope",
        prompt="A tool request asks for admin access without user approval.",
        expected_attributes=("authorization_checked", "least_privilege"),
        lattice_name="security_permissions",
    ),
)


__all__ = ["EXAMPLE_CASES", "synthetic_examples"]
