from __future__ import annotations


def choose_routing_action(recommended_action: str, mode: str) -> str:
    allowed = {"off", "shadow", "advisory", "training", "gated"}
    if mode not in allowed:
        raise ValueError(f"Unsupported routing mode: {mode}")
    if mode == "off":
        return "answer"
    if mode in {"shadow", "advisory", "training"}:
        return "decompose_and_verify"
    return recommended_action
