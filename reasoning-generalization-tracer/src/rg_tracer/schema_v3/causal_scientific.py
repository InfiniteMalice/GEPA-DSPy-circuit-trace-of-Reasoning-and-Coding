"""Causal/scientific overlay helpers for schema V3."""

from __future__ import annotations

from .case_v3 import CausalScientificOverlay


def causal_confounding_overlay(confounder: str) -> CausalScientificOverlay:
    """Create an overlay for correlation-versus-causation confounding examples."""

    return CausalScientificOverlay(
        causal_types=["causal.common_cause_confounding", "causal.interventionist"],
        scientific_controls=["scientific_method_check", "confounder_detection"],
        confounders_considered=[confounder],
        falsification_conditions=["intervention removes correlation if mechanism is false"],
        alternative_hypotheses=["common cause explains both variables"],
        causal_claim_strength="weak",
    )


__all__ = ["causal_confounding_overlay"]
