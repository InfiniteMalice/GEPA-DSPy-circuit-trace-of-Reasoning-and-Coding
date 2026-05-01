from __future__ import annotations

from .abstention_policy import is_abstention_reasonable
from .claim_extraction import extract_atomic_claims
from .config import FactualityCertificationConfig
from .constraints import compute_constraint_scores
from .evidence_matching import match_claims_to_evidence
from .integrations import to_gepa_features
from .logging_schema import make_log_bundle
from .metrics import hallucination_rate, unsupported_atomic_claim_rate
from .observability import infer_observability_tier
from .overrefusal_guard import find_scoped_alternative
from .routing import choose_routing_action
from .taxonomy import build_taxonomy
from .trace_export import build_trace_package
from .types import CertificationResult, EvidenceItem


def certify_answer(
    prompt: str,
    answer: str,
    evidence: list[EvidenceItem] | None = None,
    context: str | None = None,
    trace_summary: str | None = None,
    config: FactualityCertificationConfig | None = None,
) -> CertificationResult:
    """Certify answer factual support with optional gating and rich diagnostics."""
    cfg = config or FactualityCertificationConfig()
    if not cfg.enabled or cfg.mode == "off":
        return CertificationResult(
            "off", "certified", 0.0, 0.0, 1.0, [], "answer", revised_answer=answer
        )

    evidence = evidence or []
    claims, mapping = extract_atomic_claims(answer, max_claims=cfg.claim_extraction.max_claims)
    supports = match_claims_to_evidence(claims, evidence, cfg.evidence_matching)

    unsupported = sum(s.support_label == "unsupported" for s in supports)
    contradicted = sum(s.support_label == "contradicted" for s in supports)
    supported = sum(s.support_label == "supported" for s in supports)
    total = max(len(supports), 1)
    risk = hallucination_rate(supports)

    overall = "certified" if supported == total else "partial" if supported > 0 else "uncertified"
    action = "answer" if overall == "certified" else "answer_with_qualifications"
    if contradicted > 0:
        action = "answer_partially"
    if unsupported / total >= 0.75:
        action = "abstain"

    alt = find_scoped_alternative(prompt, answer, claims, supports, safety_context=None)
    if action in {"abstain", "refuse"} and alt.scoped_answer_possible:
        action = alt.action

    if cfg.mode != "gated" and action in {"refuse", "abstain"}:
        action = "answer_with_qualifications"

    declared_conf = 0.9 if "definitely" in answer.lower() else 0.5
    constraint_scores = compute_constraint_scores(supports, declared_conf)
    taxonomy = build_taxonomy(supported, contradicted)
    observability = infer_observability_tier(False, bool(evidence), bool(trace_summary))
    logs = make_log_bundle(
        raw_input=prompt,
        raw_answer=answer,
        atomic_fact_list=[c.text for c in claims],
        fact_verdict_per_fact=[s.support_label for s in supports],
        evidence_per_fact=[s.evidence_ids for s in supports],
        unsupported_fact_indices=[
            i for i, s in enumerate(supports) if s.support_label == "unsupported"
        ],
        contradiction_fact_indices=[
            i for i, s in enumerate(supports) if s.support_label == "contradicted"
        ],
        verification_status="weakly_checked" if evidence else "unverified",
        observability_tier=observability,
        clause_to_fact_mapping=mapping,
    )

    case_projection = {
        **to_gepa_features(
            CertificationResult(
                cfg.mode, overall, risk, 0.0, 1.0 - unsupported / total, supports, action
            )
        ),
        "confidence_score": declared_conf,
        "overrefusal_detected": action in {"abstain", "refuse"} and alt.scoped_answer_possible,
        "scoped_answer_possible": alt.scoped_answer_possible,
        "has_references": any(e.citation for e in evidence),
        "has_current_references": any(e.timestamp for e in evidence),
    }

    trace = build_trace_package({"prompt": prompt, "answer": answer, "hallucination_risk": risk})
    warnings = [] if action == "answer" else ["qualification_recommended"]
    return CertificationResult(
        mode=cfg.mode,
        overall_label=overall,
        hallucination_risk=risk,
        overrefusal_risk=1.0 if case_projection["overrefusal_detected"] else 0.0,
        useful_answer_retention_score=1.0 - unsupported_atomic_claim_rate(supports),
        claim_support=supports,
        recommended_action=action,
        revised_answer=answer,
        warnings=warnings,
        metrics={
            "constraint_scores": constraint_scores,
            "requires_current_source": any(c.requires_current_source for c in claims),
        },
        logs=logs,
        taxonomy=taxonomy,
        case_projection=case_projection,
        trace_bundle_id=trace["trace_package_id"],
    )
