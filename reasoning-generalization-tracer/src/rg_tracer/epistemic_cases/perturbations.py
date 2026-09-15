"""Explicit adapters from DSPy perturbation mechanisms to the V5 registry."""

from . import validate_coordinate

_MECHANISMS = {
    "paraphrase": ("PARAPHRASE", "REPRESENTATION_SENSITIVITY"),
    "representation_change": ("PARAPHRASE", "REPRESENTATION_SENSITIVITY"),
    "group_theoretic_paraphrase": ("PARAPHRASE", "REPRESENTATION_SENSITIVITY"),
    "translation": ("PARAPHRASE", "CROSS_LANGUAGE"),
    "cross_language": ("PARAPHRASE", "CROSS_LANGUAGE"),
    "semantic_laundering": ("REWARD_PRESSURE", "SEMANTIC_LAUNDERING"),
    "multi_turn_semantic_laundering": ("REWARD_PRESSURE", "MULTI_TURN_LAUNDERING"),
    "self_serving_reasoning": ("REWARD_PRESSURE", "SELF_SERVING_JUSTIFICATION"),
    "agent_preservation_rationalization": ("REWARD_PRESSURE", "SELF_PRESERVATION_PRESSURE"),
    "evaluator_gaming": ("REWARD_PRESSURE", "EVALUATOR_MANIPULATION"),
    "proxy_gaming": ("REWARD_PRESSURE", "REWARD_PROXY_SUBSTITUTION"),
    "tool_execution_failure": ("TOOL_ERROR", None),
    "tool_output_injection": ("TOOL_OUTPUT_INJECTION", None),
}


def stripe_for_mechanism(mechanism: str) -> tuple[str, str | None]:
    """Resolve an explicitly selected perturbation; diagnostic overlays are not inferred."""
    try:
        stripe, subtype = _MECHANISMS[mechanism]
    except KeyError as exc:
        raise ValueError(f"No registered robustness mapping for {mechanism!r}") from exc
    validate_coordinate(0, stripe, subtype)
    return stripe, subtype
