from __future__ import annotations

import re

from .types import AtomicClaim, ClaimType

_TEMPORAL_HINTS = {"today", "current", "latest", "recent", "now"}
_HEDGE_PREFIX = re.compile(r"^(?:i think|in my opinion)\s*[,:-]?\s*", re.IGNORECASE)


def extract_atomic_claims(
    answer: str, max_claims: int = 64, split_compound_claims: bool = True
) -> tuple[list[AtomicClaim], dict[int, list[str]]]:
    """Split answer into simple atomic claims without external LLM calls."""
    claims: list[AtomicClaim] = []
    mapping: dict[int, list[str]] = {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    for i, sentence in enumerate(sentences):
        if split_compound_claims:
            parts = re.split(r"\b(?:and|but)\b|[;,]", sentence)
        else:
            parts = [sentence]
        clauses = [c.strip() for c in parts if c.strip()]
        mapping[i] = clauses
        for clause in clauses:
            if len(claims) >= max_claims:
                break
            cleaned = _HEDGE_PREFIX.sub("", clause).strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            claim_type = ClaimType.FACTUAL.value
            if re.search(r"\d", cleaned):
                claim_type = ClaimType.NUMERIC.value
            if any(re.search(rf"\b{re.escape(h)}\b", lowered) for h in _TEMPORAL_HINTS):
                claim_type = ClaimType.TEMPORAL.value
            claims.append(
                AtomicClaim(
                    id=f"c{len(claims)}",
                    text=cleaned,
                    claim_type=claim_type,
                    requires_current_source=claim_type == ClaimType.TEMPORAL.value,
                    verifiability_class="directly_verifiable",
                )
            )
    return claims, mapping
