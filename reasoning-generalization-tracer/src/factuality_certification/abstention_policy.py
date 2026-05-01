from __future__ import annotations


def is_abstention_reasonable(has_evidence: bool, unsupported_ratio: float) -> bool:
    return (not has_evidence) and unsupported_ratio > 0.5
