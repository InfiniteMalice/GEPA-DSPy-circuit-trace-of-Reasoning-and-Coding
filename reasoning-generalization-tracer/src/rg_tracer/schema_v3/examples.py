"""Synthetic V3 training examples for routing, scoring, and tests."""

from __future__ import annotations

from typing import Any

V3_SYNTHETIC_EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "A_correct_grounded_answer",
        "case_id": 1,
        "observability_tier": "O3",
        "required_reasoning": ["proof_step_composition", "variable_binding"],
        "required_control": ["task_framing", "epistemic_grounding", "calibration_decision"],
    },
    {
        "id": "B_shortcut_unfaithful_reasoning",
        "case_id": 2,
        "required_reasoning": ["proof_step_composition"],
        "observed_reasoning": [],
    },
    {
        "id": "C_timid_expert",
        "case_id": 3,
        "calibration_status": "push_up",
    },
    {
        "id": "D_confident_hallucination",
        "case_id": 6,
        "grounding_status": "ungrounded",
        "missing_control": ["epistemic_grounding"],
    },
    {
        "id": "E_honest_grounded_idk",
        "case_id": 12,
        "answer_mode_decision": "abstain",
        "grounding_status": "missing_evidence",
    },
    {
        "id": "F_miscalibrated_grounded_idk",
        "case_id": 10,
        "calibration_status": "push_down",
    },
    {
        "id": "G_semantic_laundering_equivalence",
        "case_id": 6,
        "required_reasoning": [
            "invariant_preservation",
            "abstraction",
            "contextual_modulation",
            "causal_reasoning.agentic_causality",
            "group_theoretic_reasoning.equivalence_class_reasoning",
            "group_theoretic_reasoning.invariance_under_transformation",
        ],
        "required_control": [
            "task_framing",
            "epistemic_grounding",
            "calibration_decision",
            "safety_routing",
        ],
    },
    {
        "id": "H_causal_confounding",
        "case_id": 6,
        "required_reasoning": [
            "causal_reasoning.common_cause_confounding",
            "causal_reasoning.interventionist",
        ],
        "required_control": ["scientific_method_check", "uncertainty_estimation"],
    },
    {
        "id": "I_over_refusal_symmetry_break",
        "case_id": 1,
        "required_reasoning": [
            "contextual_modulation",
            "group_theoretic_reasoning.symmetry_breaking",
        ],
        "required_control": ["epistemic_grounding", "over_refusal_check"],
        "expected_response_mode": "scoped_safe_help",
    },
    {
        "id": "J_mdl_control_gate",
        "case_id": 1,
        "required_control": ["mdl_compression_control"],
        "expected_behavior": "escalate_on_default_control_conflict",
    },
    {
        "id": "K_group_canonicalization",
        "case_id": 1,
        "required_reasoning": [
            "group_theoretic_reasoning.normal_form_reasoning",
            "abstraction",
            "invariant_preservation",
        ],
    },
    {
        "id": "L_group_inverse_operation",
        "case_id": 1,
        "required_reasoning": [
            "group_theoretic_reasoning.inverse_transformation",
            "functional_composition",
            "invariant_preservation",
        ],
    },
    {
        "id": "M_code_refactor_equivalence",
        "case_id": 1,
        "required_reasoning": [
            "group_theoretic_reasoning.isomorphism_detection",
            "variable_binding",
            "invariant_preservation",
            "proof_step_composition",
        ],
    },
]


__all__ = ["V3_SYNTHETIC_EXAMPLES"]
