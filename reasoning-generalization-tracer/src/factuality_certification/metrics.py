from __future__ import annotations

from .types import ClaimSupport


def hallucination_rate(supports: list[ClaimSupport]) -> float:
    """Fraction of contradicted/unsupported claims."""
    bad = sum(s.support_label in {"unsupported", "contradicted"} for s in supports)
    return bad / max(len(supports), 1)


def unsupported_atomic_claim_rate(supports: list[ClaimSupport]) -> float:
    return sum(s.support_label == "unsupported" for s in supports) / max(len(supports), 1)


def contradicted_claim_rate(supports: list[ClaimSupport]) -> float:
    return sum(s.support_label == "contradicted" for s in supports) / max(len(supports), 1)


def overrefusal_rate(overrefusal_cases: int, answerable_cases: int) -> float:
    return overrefusal_cases / max(answerable_cases, 1)
