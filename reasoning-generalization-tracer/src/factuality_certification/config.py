from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ClaimExtractionConfig:
    enabled: bool = True
    max_claims: int = 64
    split_compound_claims: bool = True


@dataclass
class EvidenceMatchingConfig:
    enabled: bool = True
    min_support_score: float = 0.65
    contradiction_threshold: float = 0.60
    source_quality_weight: float = 0.25
    retrieval_weight: float = 0.35
    entailment_weight: float = 0.40

    def __post_init__(self) -> None:
        for name in ["min_support_score", "contradiction_threshold"]:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"EvidenceMatchingConfig.{name} must be in [0.0, 1.0], got {value}"
                )

        weights = ["source_quality_weight", "retrieval_weight", "entailment_weight"]
        vals = [getattr(self, w) for w in weights]
        if any(v < 0.0 or v > 1.0 for v in vals):
            raise ValueError("EvidenceMatchingConfig weights must be in [0.0, 1.0]")
        total = sum(vals)
        if total <= 0.0:
            raise ValueError("EvidenceMatchingConfig weights must sum to > 0.0")
        if abs(total - 1.0) > 1e-9:
            self.source_quality_weight = self.source_quality_weight / total
            self.retrieval_weight = self.retrieval_weight / total
            self.entailment_weight = self.entailment_weight / total


@dataclass
class CertificationThresholdConfig:
    certified_threshold: float = 0.80
    partial_threshold: float = 0.55
    abstention_threshold: float = 0.35
    refusal_threshold: float = 0.85
    require_scope_before_refusal: bool = True
    allow_partial_answers: bool = True
    allow_uncertainty_qualified_answers: bool = True

    def __post_init__(self) -> None:
        if not (
            0.0
            <= self.abstention_threshold
            < self.partial_threshold
            < self.certified_threshold
            < self.refusal_threshold
            <= 1.0
        ):
            raise ValueError(
                "CertificationThresholdConfig requires "
                "0.0 <= abstention_threshold < partial_threshold < "
                "certified_threshold < refusal_threshold <= 1.0; got "
                f"abstention_threshold={self.abstention_threshold}, "
                f"partial_threshold={self.partial_threshold}, "
                f"certified_threshold={self.certified_threshold}, "
                f"refusal_threshold={self.refusal_threshold}"
            )


@dataclass
class OverRefusalGuardConfig:
    enabled: bool = True
    safe_scoped_answer_preferred: bool = True
    ask_clarifying_before_refusal_when_appropriate: bool = True


@dataclass
class LoggingConfig:
    log_atomic_claims: bool = True
    log_evidence_matches: bool = True
    log_constraint_scores: bool = True
    log_counterfactual_action: bool = True
    log_for_attribution_graphs: bool = True
    log_taxonomy_labels: bool = True
    log_guessing_pressure: bool = True
    log_related_query_consistency: bool = True


@dataclass
class TrainingConfig:
    positive_only_trace_rewards: bool = True
    export_reward_features: bool = True


@dataclass
class FactualityCertificationConfig:
    enabled: bool = True
    mode: Literal["off", "shadow", "advisory", "gated", "training"] = "shadow"
    claim_extraction: ClaimExtractionConfig = field(default_factory=ClaimExtractionConfig)
    evidence_matching: EvidenceMatchingConfig = field(default_factory=EvidenceMatchingConfig)
    certification: CertificationThresholdConfig = field(
        default_factory=CertificationThresholdConfig
    )
    overrefusal_guard: OverRefusalGuardConfig = field(default_factory=OverRefusalGuardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        allowed = {"off", "shadow", "advisory", "gated", "training"}
        if self.mode not in allowed:
            raise ValueError(
                "FactualityCertificationConfig.mode must be one of "
                f"{sorted(allowed)}, got {self.mode}"
            )
