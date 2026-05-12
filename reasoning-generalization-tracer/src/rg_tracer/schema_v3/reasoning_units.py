"""Reasoning-unit registry for the 13-case schema V3 overlay."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

CAUSAL_SUBTYPES = (
    "temporal_order",
    "mechanism",
    "counterfactual_dependence",
    "interventionist",
    "common_cause_confounding",
    "necessary_vs_sufficient",
    "enabling_vs_triggering",
    "overdetermination",
    "probabilistic_causality",
    "dose_response",
    "level_shifting",
    "material_formal_efficient_final",
    "agentic_causality",
    "normative_responsibility",
)

GROUP_THEORETIC_SUBTYPES = (
    "identity_transformation",
    "inverse_transformation",
    "closure_under_operation",
    "associativity_of_composition",
    "symmetry_detection",
    "invariance_under_transformation",
    "equivalence_class_reasoning",
    "orbit_reasoning",
    "stabilizer_reasoning",
    "quotient_structure_reasoning",
    "normal_form_reasoning",
    "isomorphism_detection",
    "symmetry_breaking",
    "conservation_law_analogy",
)

REASONING_FAMILIES = (
    "recursive",
    "functional_composition",
    "type_constrained_composition",
    "abstraction",
    "instantiation",
    "variable_binding",
    "relational_composition",
    "constraint_composition",
    "decomposition",
    "proof_step_composition",
    "analogy",
    "invariant_preservation",
    "contextual_modulation",
    "causal_reasoning",
    "hierarchical_composition",
    "dialectical_composition",
    "compression_expansion",
    "group_theoretic_reasoning",
)


@dataclass(frozen=True)
class ReasoningUnitEntry:
    """Machine-readable reasoning-unit registry entry."""

    name: str
    display_name: str
    definition: str
    input_type: tuple[str, ...]
    output_type: tuple[str, ...]
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    minimal_tests: tuple[str, ...] = field(default_factory=tuple)
    transfer_tests: tuple[str, ...] = field(default_factory=tuple)
    failure_modes: tuple[str, ...] = field(default_factory=tuple)
    composition_partners: tuple[str, ...] = field(default_factory=tuple)
    example_domains: tuple[str, ...] = field(default_factory=tuple)
    subtypes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Convert sequence fields to tuples so frozen entries are immutable."""

        for name in (
            "input_type",
            "output_type",
            "dependencies",
            "minimal_tests",
            "transfer_tests",
            "failure_modes",
            "composition_partners",
            "example_domains",
            "subtypes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "name": self.name,
            "display_name": self.display_name,
            "definition": self.definition,
            "input_type": list(self.input_type),
            "output_type": list(self.output_type),
            "dependencies": list(self.dependencies),
            "minimal_tests": list(self.minimal_tests),
            "transfer_tests": list(self.transfer_tests),
            "failure_modes": list(self.failure_modes),
            "composition_partners": list(self.composition_partners),
            "example_domains": list(self.example_domains),
            "subtypes": list(self.subtypes),
        }


def _basic_entry(name: str) -> ReasoningUnitEntry:
    display = name.replace("_", " ").title()
    return ReasoningUnitEntry(
        name=name,
        display_name=display,
        definition=f"Use {display.lower()} as a public, auditable reasoning unit.",
        input_type=("task_context", "candidate_answer"),
        output_type=("unit_trace", "diagnostic_tags"),
        dependencies=(),
        minimal_tests=(f"{name} applies to a simple benchmark example",),
        transfer_tests=(f"{name} transfers across natural-language and code tasks",),
        failure_modes=("unit_missing", "unit_misapplied"),
        composition_partners=("abstraction", "constraint_composition"),
        example_domains=("math", "natural_language", "code"),
    )


def _causal_entry() -> ReasoningUnitEntry:
    return ReasoningUnitEntry(
        name="causal_reasoning",
        display_name="Causal Reasoning",
        definition=(
            "Reason about temporal order, mechanisms, interventions, confounders, "
            "counterfactual dependence, and causal claim strength."
        ),
        input_type=["claim", "evidence", "candidate_mechanism"],
        output_type=["causal_type", "confounders", "claim_strength"],
        dependencies=["relational_composition", "constraint_composition"],
        minimal_tests=["correlation is not treated as causation"],
        transfer_tests=["scientific study critique", "policy impact analysis"],
        failure_modes=["post_hoc_fallacy", "missed_confounder", "overclaimed_causality"],
        composition_partners=["scientific_method_check", "contextual_modulation"],
        example_domains=["science", "medicine", "policy", "causal_safety"],
        subtypes=CAUSAL_SUBTYPES,
    )


def _group_entry() -> ReasoningUnitEntry:
    return ReasoningUnitEntry(
        name="group_theoretic_reasoning",
        display_name="Group-Theoretic Reasoning",
        definition=(
            "Reason over transformations, symmetries, invariants, equivalence "
            "classes, reversible operations, canonical forms, and "
            "structure-preserving mappings. This family helps detect when "
            "surface changes do or do not alter the relevant underlying structure."
        ),
        input_type=[
            "object_or_representation",
            "transformation_or_set_of_transformations",
            "target_property",
        ],
        output_type=[
            "invariant_properties",
            "changed_properties",
            "equivalence_class",
            "canonical_form",
            "symmetry_breaks",
            "reachable_variants",
        ],
        dependencies=[
            "invariant_preservation",
            "abstraction",
            "relational_composition",
            "type_constrained_composition",
            "analogy",
        ],
        minimal_tests=[
            "variable renaming preserves equation validity",
            "translation preserves factual claim",
            "paraphrase preserves user intent",
            "rotation preserves shape identity",
            "inverse operation restores original state",
        ],
        transfer_tests=[
            "math transformation invariance",
            "semantic laundering detection",
            "code refactor behavior preservation",
            "legal reformatting obligation preservation",
            "causal affordance under euphemism",
        ],
        failure_modes=[
            "surface_form_bias",
            "false_equivalence",
            "missed_symmetry_break",
            "treating irreversible operations as reversible",
            "failing to canonicalize variants",
            "confusing same topic with same intent",
        ],
        composition_partners=[
            "invariant_preservation",
            "abstraction",
            "instantiation",
            "analogy",
            "contextual_modulation",
            "causal_reasoning",
            "proof_step_composition",
            "constraint_composition",
        ],
        example_domains=[
            "arithmetic",
            "algebra",
            "natural_language",
            "safety",
            "legal_reasoning",
            "code_refactoring",
            "mechanistic_interpretability",
        ],
        subtypes=GROUP_THEORETIC_SUBTYPES,
    )


def build_reasoning_unit_registry() -> Mapping[str, ReasoningUnitEntry]:
    """Build the V3 reasoning-unit registry."""

    entries = {name: _basic_entry(name) for name in REASONING_FAMILIES}
    entries["causal_reasoning"] = _causal_entry()
    entries["group_theoretic_reasoning"] = _group_entry()
    return MappingProxyType(entries)


REASONING_UNIT_REGISTRY = build_reasoning_unit_registry()


__all__ = [
    "CAUSAL_SUBTYPES",
    "GROUP_THEORETIC_SUBTYPES",
    "REASONING_FAMILIES",
    "REASONING_UNIT_REGISTRY",
    "ReasoningUnitEntry",
    "build_reasoning_unit_registry",
]
