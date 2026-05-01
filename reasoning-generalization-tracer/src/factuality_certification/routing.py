from __future__ import annotations


def choose_routing_action(recommended_action: str, mode: str) -> str:
    if mode == "off":
        return "answer"
    if mode in {"shadow", "advisory", "training"}:
        return "decompose_and_verify"
    if mode == "gated":
        return recommended_action
    return recommended_action
