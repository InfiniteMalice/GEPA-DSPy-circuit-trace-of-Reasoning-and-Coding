from __future__ import annotations

import re

from .config import EvidenceMatchingConfig
from .types import AtomicClaim, ClaimSupport, EvidenceItem


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _symbol_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"\b[a-zA-Z]\b", text.lower()) if token not in {"a", "i"}}


def match_claims_to_evidence(
    claims: list[AtomicClaim], evidence: list[EvidenceItem], config: EvidenceMatchingConfig
) -> list[ClaimSupport]:
    supports: list[ClaimSupport] = []
    for claim in claims:
        c_tokens = _tokens(claim.text)
        c_symbols = _symbol_tokens(claim.text)
        best_score = 0.0
        best_eids: list[str] = []
        contradiction = 0.0
        contradiction_eids: list[str] = []
        for item in evidence:
            e_tokens = _tokens(item.text)
            overlap = len(c_tokens & e_tokens) / max(len(c_tokens), 1)
            e_symbols = _symbol_tokens(item.text)
            symbol_mismatch = bool(c_symbols or e_symbols) and c_symbols != e_symbols
            if symbol_mismatch:
                overlap = min(overlap, 0.5)
            weighted = overlap * config.entailment_weight
            weighted += item.quality_score * config.source_quality_weight
            weighted += item.retrieval_score * config.retrieval_weight
            if symbol_mismatch:
                weighted = min(weighted, config.min_support_score * 0.9)
            if weighted > best_score:
                best_score = weighted
                best_eids = [item.id]
            if (" not " in f" {claim.text.lower()} ") ^ (" not " in f" {item.text.lower()} "):
                if overlap >= 0.2:
                    contradiction = max(contradiction, 0.8)
                    contradiction_eids.append(item.id)
        if contradiction >= config.contradiction_threshold:
            label = "contradicted"
        elif best_score >= config.min_support_score:
            label = "supported"
        elif best_score >= config.min_support_score * 0.6:
            label = "partially_supported"
        else:
            label = "unsupported"
        evidence_ids = contradiction_eids if label == "contradicted" else best_eids
        supports.append(
            ClaimSupport(
                claim_id=claim.id,
                support_label=label,
                support_score=min(best_score, 1.0),
                contradiction_score=contradiction,
                evidence_ids=evidence_ids,
                rationale=f"lexical_weighted={best_score:.3f}",
                needs_abstention=label in {"unsupported", "contradicted"},
                needs_qualification=label in {"unsupported", "partially_supported", "contradicted"},
            )
        )
    return supports
