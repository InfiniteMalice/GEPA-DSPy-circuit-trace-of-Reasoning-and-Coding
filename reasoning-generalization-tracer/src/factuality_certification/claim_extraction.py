from __future__ import annotations

import re

from .types import AtomicClaim, ClaimType

_TEMPORAL_HINTS = {"today", "current", "latest", "recent", "now"}


def extract_atomic_claims(
    answer: str, max_claims: int = 64
) -> tuple[list[AtomicClaim], dict[int, list[str]]]:
    """Split answer into simple atomic claims without external LLM calls."""
    claims: list[AtomicClaim] = []
    mapping: dict[int, list[str]] = {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    for i, sentence in enumerate(sentences):
        clauses = [c.strip() for c in re.split(r"\b(?:and|but|;|,)\b", sentence) if c.strip()]
        mapping[i] = clauses
        for clause in clauses:
            if len(claims) >= max_claims:
                break
            lowered = clause.lower()
            if any(w in lowered for w in ["i think", "in my opinion"]) and not re.search(
                r"\d", clause
            ):
                continue
            claim_type = ClaimType.FACTUAL.value
            if re.search(r"\d", clause):
                claim_type = ClaimType.NUMERIC.value
            if any(h in lowered for h in _TEMPORAL_HINTS):
                claim_type = ClaimType.TEMPORAL.value
            claims.append(
                AtomicClaim(
                    id=f"c{len(claims)}",
                    text=clause,
                    claim_type=claim_type,
                    requires_current_source=claim_type == ClaimType.TEMPORAL.value,
                    verifiability_class="directly_verifiable",
                )
            )
    return claims, mapping
