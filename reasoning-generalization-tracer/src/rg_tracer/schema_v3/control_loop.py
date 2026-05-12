"""Control-loop operation registry for schema V3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CONTROL_OPERATIONS = [
    "task_framing",
    "epistemic_grounding",
    "method_selection",
    "reasoning_unit_selection",
    "assumption_tracking",
    "uncertainty_estimation",
    "calibration_decision",
    "consistency_checking",
    "error_localization",
    "revision_control",
    "scientific_method_check",
    "epistemic_boundary_abstention",
    "mdl_compression_control",
]

SCIENTIFIC_METHOD_CHECKS = [
    "hypothesis_formation",
    "operationalization",
    "prediction",
    "falsification_condition",
    "control_group",
    "confounder_detection",
    "measurement_validity",
    "replication_check",
    "effect_size_reasoning",
    "alternative_hypothesis_comparison",
]

MDL_COMPRESSION_CHECKS = [
    "fast_default_answer",
    "controlled_deliberative_answer",
    "disagreement_conflict_signal",
    "escalation_rule",
    "compression_candidate_rule",
    "unsafe_overcompression_guardrails",
]


@dataclass(frozen=True)
class ControlLoopEntry:
    """Machine-readable control-loop registry entry."""

    name: str
    display_name: str
    definition: str
    inputs: list[str]
    outputs: list[str]
    failure_modes: list[str] = field(default_factory=list)
    example_domains: list[str] = field(default_factory=list)
    composition_partners: list[str] = field(default_factory=list)
    scoring_notes: str = ""
    subchecks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "name": self.name,
            "display_name": self.display_name,
            "definition": self.definition,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "failure_modes": list(self.failure_modes),
            "example_domains": list(self.example_domains),
            "composition_partners": list(self.composition_partners),
            "scoring_notes": self.scoring_notes,
            "subchecks": list(self.subchecks),
        }


def _entry(name: str) -> ControlLoopEntry:
    display = name.replace("_", " ").title()
    return ControlLoopEntry(
        name=name,
        display_name=display,
        definition=f"Invoke {display.lower()} as a public control operation.",
        inputs=["task_context", "risk_context", "candidate_answer"],
        outputs=["control_decision", "public_diagnostics"],
        failure_modes=["control_not_invoked", "control_misapplied"],
        example_domains=["qa", "safety", "science", "code"],
        composition_partners=["calibration_decision", "epistemic_grounding"],
        scoring_notes=(
            "Positive when observed and required; zero when absent unless behavior " "overclaims."
        ),
    )


def build_control_loop_registry() -> dict[str, ControlLoopEntry]:
    """Build the V3 control-loop registry."""

    entries = {name: _entry(name) for name in CONTROL_OPERATIONS}
    entries["scientific_method_check"] = ControlLoopEntry(
        name="scientific_method_check",
        display_name="Scientific Method Check",
        definition=(
            "Check claims against scientific controls before accepting causal " "conclusions."
        ),
        inputs=["hypothesis", "measurements", "candidate_causal_claim"],
        outputs=["falsification_conditions", "confounders", "claim_strength"],
        failure_modes=["missed_confounder", "unfalsifiable_claim", "invalid_measurement"],
        example_domains=["science", "medicine", "social_science"],
        composition_partners=["causal_reasoning", "uncertainty_estimation"],
        scoring_notes=(
            "Reward observed scientific controls when causal/scientific claims " "require them."
        ),
        subchecks=SCIENTIFIC_METHOD_CHECKS,
    )
    entries["mdl_compression_control"] = ControlLoopEntry(
        name="mdl_compression_control",
        display_name="MDL Compression Control",
        definition=(
            "Compare fast/default and controlled answers; escalate when conflict, "
            "grounding gaps, or causal checks make compression unsafe."
        ),
        inputs=["default_answer", "controlled_answer", "grounding_status"],
        outputs=["conflict_signal", "escalation_decision", "compression_guardrails"],
        failure_modes=["unsafe_overcompression", "missed_conflict", "unneeded_escalation"],
        example_domains=["qa", "code", "science", "safety"],
        composition_partners=["method_selection", "calibration_decision"],
        scoring_notes="Reward escalation when required and safe compression only with guardrails.",
        subchecks=MDL_COMPRESSION_CHECKS,
    )
    return entries


CONTROL_LOOP_REGISTRY = build_control_loop_registry()


__all__ = [
    "CONTROL_LOOP_REGISTRY",
    "CONTROL_OPERATIONS",
    "ControlLoopEntry",
    "MDL_COMPRESSION_CHECKS",
    "SCIENTIFIC_METHOD_CHECKS",
    "build_control_loop_registry",
]
