from __future__ import annotations

from .types import CertificationResult


def to_gepa_features(result: CertificationResult) -> dict:
    return {
        "is_supported": result.overall_label in {"certified", "partial"},
        "is_contradicted": any(s.support_label == "contradicted" for s in result.claim_support),
        "abstained": result.recommended_action == "abstain",
        "refused": result.recommended_action == "refuse",
        "final_answer_behavior_label": result.recommended_action,
    }


def reward_features(result: CertificationResult) -> dict[str, float]:
    positive = {
        "identified_uncertainty": 1.0 if "qualification" in " ".join(result.warnings) else 0.0,
        "used_supporting_evidence": (
            1.0 if any(s.evidence_ids for s in result.claim_support) else 0.0
        ),
        "chose_calibrated_abstention": 1.0 if result.recommended_action == "abstain" else 0.0,
        "avoided_overrefusal": 1.0 if result.overrefusal_risk < 0.5 else 0.0,
        "produced_trace_package": 1.0 if result.trace_bundle_id else 0.0,
    }
    return {k: max(v, 0.0) for k, v in positive.items()}
