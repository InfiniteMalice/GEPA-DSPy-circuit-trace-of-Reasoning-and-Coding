"""MDL-control helpers for schema V3."""

from __future__ import annotations

from .case_v3 import MDLControlOverlay


def build_mdl_control_overlay(
    *,
    default_answer: str | None,
    controlled_answer: str | None,
    grounding_status: str = "unknown",
    causal_check_failed: bool = False,
    guardrails: list[str] | None = None,
) -> MDLControlOverlay:
    """Build an MDL overlay and require escalation on conflict or failed checks."""

    conflict = bool(default_answer and controlled_answer and default_answer != controlled_answer)
    missing_grounding = grounding_status in {"ungrounded", "missing_evidence"}
    escalation_required = conflict or missing_grounding or causal_check_failed
    return MDLControlOverlay(
        default_answer=default_answer,
        controlled_answer=controlled_answer,
        default_control_conflict=conflict,
        escalation_required=escalation_required,
        escalation_taken=bool(escalation_required and controlled_answer),
        compression_candidate=bool(not escalation_required and default_answer == controlled_answer),
        compression_guardrails=list(guardrails or []),
    )


__all__ = ["build_mdl_control_overlay"]
