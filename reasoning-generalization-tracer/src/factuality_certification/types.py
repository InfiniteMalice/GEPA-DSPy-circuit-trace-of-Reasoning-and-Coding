from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ClaimType(str, Enum):
    FACTUAL = "factual"
    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    LEGAL = "legal"
    MEDICAL = "medical"
    SCIENTIFIC = "scientific"
    CITATION_REQUIRED = "citation_required"
    SUBJECTIVE = "subjective"
    INSTRUCTION = "instruction"
    SAFETY_RELEVANT = "safety_relevant"


@dataclass
class AtomicClaim:
    id: str
    text: str
    claim_type: str = ClaimType.FACTUAL.value
    importance: float = 0.5
    requires_current_source: bool = False
    source_span: str | None = None
    answer_span: str | None = None
    hallucination_primary_type: str | None = None
    verifiability_class: str | None = None


@dataclass
class EvidenceItem:
    id: str
    text: str
    source: str | None = None
    citation: str | None = None
    timestamp: str | None = None
    quality_score: float = 0.5
    retrieval_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimSupport:
    claim_id: str
    support_label: str
    support_score: float
    contradiction_score: float
    evidence_ids: list[str]
    rationale: str
    needs_abstention: bool
    needs_qualification: bool


@dataclass
class CertificationResult:
    mode: str
    overall_label: str
    hallucination_risk: float
    overrefusal_risk: float
    useful_answer_retention_score: float
    claim_support: list[ClaimSupport]
    recommended_action: str
    revised_answer: str | None = None
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)
    taxonomy: dict[str, Any] = field(default_factory=dict)
    case_projection: dict[str, Any] = field(default_factory=dict)
    trace_bundle_id: str | None = None
