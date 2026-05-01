from __future__ import annotations

from dataclasses import dataclass

from .types import AtomicClaim, ClaimSupport


@dataclass
class ScopedAlternative:
    scoped_answer_possible: bool
    action: str
    reason: str


def find_scoped_alternative(
    prompt: str,
    answer: str,
    claims: list[AtomicClaim],
    supports: list[ClaimSupport],
    safety_context: dict | None = None,
) -> ScopedAlternative:
    supported = any(s.support_label in {"supported", "partially_supported"} for s in supports)
    if safety_context and safety_context.get("unsafe", False):
        return ScopedAlternative(False, "refuse", "safety-policy-required")
    if supported:
        return ScopedAlternative(True, "answer_partially", "supported-subset-available")
    if claims:
        return ScopedAlternative(
            True, "answer_with_qualifications", "uncertainty-qualification-possible"
        )
    return ScopedAlternative(False, "abstain", "no-claim-level-support")
