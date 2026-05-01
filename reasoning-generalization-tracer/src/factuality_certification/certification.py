from __future__ import annotations

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


def _declared_confidence(answer: str) -> float:
    lowered = answer.lower()
    if any(tok in lowered for tok in ["definitely", "certainly", "always"]):
        return 0.9
    if any(tok in lowered for tok in ["maybe", "might", "uncertain", "not sure"]):
        return 0.3
    return 0.6


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
    mapping: dict[int, list[str]] = {}
    if cfg.claim_extraction.enabled:
        claims, mapping = extract_atomic_claims(
            answer,
            max_claims=cfg.claim_extraction.max_claims,
            split_compound_claims=cfg.claim_extraction.split_compound_claims,
        )
        supports = (
            match_claims_to_evidence(claims, evidence, cfg.evidence_matching)
            if cfg.evidence_matching.enabled
            else []
        )
    else:
        claims, supports = [], []

    unsupported = sum(s.support_label == "unsupported" for s in supports)
    contradicted = sum(s.support_label == "contradicted" for s in supports)
    supported = sum(s.support_label == "supported" for s in supports)
    partial = sum(s.support_label == "partially_supported" for s in supports)
    total = len(supports)
    total_safe = max(total, 1)
    support_ratio = (supported + 0.5 * partial) / total_safe
    contradiction_ratio = contradicted / total_safe
    unsupported_ratio = unsupported / total_safe
    risk = hallucination_rate(supports)

    th = cfg.certification
    contradiction_trigger = contradiction_ratio >= (1.0 - th.certified_threshold)
    if total == 0:
        overall = "uncertified"
    elif contradiction_trigger and (
        unsupported_ratio >= th.refusal_threshold or contradiction_ratio >= th.refusal_threshold
    ):
        overall = "should_refuse" if cfg.mode == "gated" else "uncertified"
    elif contradiction_trigger and (
        unsupported_ratio >= th.abstention_threshold
        or contradiction_ratio >= th.abstention_threshold
    ):
        overall = "should_abstain" if cfg.mode == "gated" else "uncertified"
    elif support_ratio >= th.certified_threshold and contradiction_ratio < (
        1.0 - th.certified_threshold
    ):
        overall = "certified"
    elif support_ratio >= th.partial_threshold:
        overall = "partial"
    elif unsupported_ratio >= th.refusal_threshold:
        overall = "should_refuse" if cfg.mode == "gated" else "uncertified"
    elif unsupported_ratio >= th.abstention_threshold:
        overall = "should_abstain" if cfg.mode == "gated" else "uncertified"
    else:
        overall = "uncertified"

    if overall == "certified":
        action = "answer"
    elif unsupported_ratio >= th.refusal_threshold or contradiction_ratio >= th.refusal_threshold:
        action = "refuse"
    elif (
        unsupported_ratio >= th.abstention_threshold
        or contradiction_ratio >= th.abstention_threshold
    ):
        action = "abstain"
    elif (
        th.allow_partial_answers
        and support_ratio > 0
        and contradiction_ratio < th.refusal_threshold
        and (contradiction_ratio >= th.partial_threshold or support_ratio >= th.partial_threshold)
    ):
        action = "answer_partially"
    elif (
        th.allow_uncertainty_qualified_answers
        and support_ratio > 0
        and contradiction_ratio < th.refusal_threshold
    ):
        action = "answer_with_qualifications"
    else:
        action = "refuse"

    alt_scoped = False
    if cfg.overrefusal_guard.enabled:
        alt = find_scoped_alternative(prompt, answer, claims, supports, safety_context=None)
        alt_scoped = alt.scoped_answer_possible
        if action == "refuse" and th.require_scope_before_refusal and alt.scoped_answer_possible:
            action = alt.action
        if action == "abstain" and alt.scoped_answer_possible and th.allow_partial_answers:
            action = alt.action

    if cfg.mode != "gated" and action in {"refuse", "abstain"}:
        if th.allow_uncertainty_qualified_answers and (
            cfg.mode == "gated" or cfg.overrefusal_guard.enabled or alt_scoped
        ):
            action = "answer_with_qualifications"

    declared_conf = _declared_confidence(answer)
    constraint_scores = compute_constraint_scores(supports, declared_conf)
    taxonomy = build_taxonomy(
        supported=supported, contradicted=contradicted, unsupported=unsupported
    )
    observability = infer_observability_tier(False, bool(evidence), bool(trace_summary))

    log_payload: dict = {
        "raw_input": prompt,
        "raw_answer": answer,
        "observability_tier": observability,
    }
    if cfg.logging.log_atomic_claims:
        log_payload["atomic_fact_list"] = [c.text for c in claims]
        log_payload["clause_to_fact_mapping"] = mapping
    if cfg.logging.log_evidence_matches:
        log_payload["fact_verdict_per_fact"] = [s.support_label for s in supports]
        log_payload["evidence_per_fact"] = [s.evidence_ids for s in supports]
        log_payload["unsupported_fact_indices"] = [
            i for i, s in enumerate(supports) if s.support_label == "unsupported"
        ]
        log_payload["contradiction_fact_indices"] = [
            i for i, s in enumerate(supports) if s.support_label == "contradicted"
        ]
    logs = make_log_bundle(**log_payload)

    case_projection = {
        **to_gepa_features(
            CertificationResult(
                cfg.mode,
                overall,
                risk,
                0.0,
                1.0 - unsupported_ratio,
                supports,
                action,
            )
        ),
        "confidence_score": declared_conf,
        "overrefusal_detected": action in {"abstain", "refuse"} and alt_scoped,
        "scoped_answer_possible": alt_scoped,
        "has_references": any(e.citation for e in evidence),
        "has_current_references": any(e.timestamp for e in evidence),
    }

    route = choose_routing_action(action, cfg.mode)
    trace = build_trace_package(
        {
            "prompt": prompt,
            "answer": answer,
            "hallucination_risk": risk,
            "routing_path": route,
        }
    )
    warnings: list[str] = []
    if action == "answer_with_qualifications":
        warnings.append("qualification_recommended")
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
            "routing_action": route,
        },
        logs=logs,
        taxonomy=taxonomy,
        case_projection=case_projection,
        trace_bundle_id=trace["trace_package_id"],
    )
