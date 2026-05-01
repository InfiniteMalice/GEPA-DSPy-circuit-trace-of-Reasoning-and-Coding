from __future__ import annotations

from .types import ClaimSupport


def compute_constraint_scores(
    supports: list[ClaimSupport], declared_confidence: float
) -> dict[str, float]:
    total = max(len(supports), 1)
    supported = sum(s.support_label == "supported" for s in supports)
    contradicted = sum(s.support_label == "contradicted" for s in supports)
    support_strength = supported / total
    contradiction = contradicted / total
    conf_gap = max(0.0, declared_confidence - support_strength)
    return {
        "C_support": support_strength,
        "C_contradiction": 1.0 - contradiction,
        "C_confidence": 1.0 - conf_gap,
        "C_non_overrefusal": 1.0,
    }
