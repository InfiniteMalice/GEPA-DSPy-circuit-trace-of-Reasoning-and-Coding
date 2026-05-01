import pytest

from factuality_certification import EvidenceItem, FactualityCertificationConfig, certify_answer
from factuality_certification.config import CertificationThresholdConfig, EvidenceMatchingConfig
from factuality_certification.integrations import reward_features
from factuality_certification.metrics import overrefusal_rate
from factuality_certification.routing import choose_routing_action
from factuality_certification.claim_extraction import extract_atomic_claims


def test_supported_claim():
    ev = [
        EvidenceItem(
            id="e1", text="Paris is the capital of France.", quality_score=1.0, retrieval_score=1.0
        )
    ]
    res = certify_answer("Q", "Paris is the capital of France.", evidence=ev)
    assert res.overall_label in {"certified", "partial"}
    assert res.recommended_action == "answer"


def test_unsupported_claim():
    res = certify_answer("Q", "Atlantis is in Europe.", evidence=[])
    assert res.overall_label == "uncertified"
    assert res.recommended_action in {"answer_with_qualifications", "abstain"}


def test_contradicted_claim_uses_contradiction_evidence_id():
    ev = [
        EvidenceItem(
            id="e1", text="Paris is the capital of France.", quality_score=0.4, retrieval_score=0.4
        ),
        EvidenceItem(
            id="e2",
            text="Paris is not the capital of France.",
            quality_score=1.0,
            retrieval_score=1.0,
        ),
    ]
    res = certify_answer("Q", "Paris is the capital of France.", evidence=ev)
    contradicted = [s for s in res.claim_support if s.support_label == "contradicted"]
    assert contradicted and "e2" in contradicted[0].evidence_ids


def test_overrefusal_guard_toggle():
    ev = [
        EvidenceItem(id="e1", text="Water boils at 100 C.", quality_score=1.0, retrieval_score=1.0)
    ]
    cfg = FactualityCertificationConfig(mode="gated")
    cfg.certification.allow_partial_answers = False
    cfg.certification.allow_uncertainty_qualified_answers = False
    cfg.overrefusal_guard.enabled = False
    res = certify_answer("Q", "Unknown.", evidence=ev, config=cfg)
    assert res.recommended_action in {"abstain", "refuse"}


def test_current_source_detection():
    ev = [
        EvidenceItem(id="e1", text="Latest price is 100.", quality_score=0.8, retrieval_score=0.9)
    ]
    res = certify_answer("current price?", "The latest price is 100.", evidence=ev)
    assert res.metrics["requires_current_source"] is True


def test_no_thought_penalty():
    res = certify_answer("Q", "unknown claim", evidence=[])
    feats = reward_features(res)
    assert all(v >= 0.0 for v in feats.values())


def test_modes_do_not_crash():
    for mode in ["off", "shadow", "advisory", "gated", "training"]:
        cfg = FactualityCertificationConfig(mode=mode, enabled=True)
        res = certify_answer("Q", "A fact.", evidence=[], config=cfg)
        assert res.mode == mode


def test_claim_extraction_disable_skips_claims():
    cfg = FactualityCertificationConfig(mode="shadow")
    cfg.claim_extraction.enabled = False
    res = certify_answer("Q", "A and B, C.", evidence=[], config=cfg)
    assert res.claim_support == []


def test_split_compound_toggle_changes_fact_count():
    cfg = FactualityCertificationConfig(mode="shadow")
    cfg.claim_extraction.split_compound_claims = False
    cfg.logging.log_atomic_claims = True
    res = certify_answer("Q", "A and B, C.", evidence=[], config=cfg)
    assert len(res.logs.get("atomic_fact_list", [])) == 1


def test_gated_flag_combinations():
    ev = [EvidenceItem(id="e1", text="Fact A is true.", quality_score=1.0, retrieval_score=1.0)]
    cfg = FactualityCertificationConfig(mode="gated")
    cfg.certification.allow_partial_answers = True
    cfg.certification.allow_uncertainty_qualified_answers = True
    cfg.certification.require_scope_before_refusal = True
    res = certify_answer("Q", "Fact A is true and Fact B is true.", evidence=ev, config=cfg)
    assert res.recommended_action in {"answer_partially", "answer_with_qualifications"}


def test_config_validation_and_routing_validation():
    with pytest.raises(ValueError):
        EvidenceMatchingConfig(min_support_score=1.2)
    with pytest.raises(ValueError):
        CertificationThresholdConfig(certified_threshold=0.2, partial_threshold=0.5)
    with pytest.raises(ValueError):
        FactualityCertificationConfig(mode="strict")
    with pytest.raises(ValueError):
        choose_routing_action("answer", "typo")


def test_overrefusal_rate_clamped():
    assert overrefusal_rate(1, 0) == 0.0
    assert overrefusal_rate(5, 2) == 1.0


def test_temporal_hint_uses_word_boundaries():
    res = certify_answer("Q", "This is currentlyness only.", evidence=[])
    assert res.metrics["requires_current_source"] is False


def test_abstain_before_qualification_when_threshold_hit():
    cfg = FactualityCertificationConfig(mode="gated")
    cfg.certification.allow_partial_answers = False
    cfg.certification.allow_uncertainty_qualified_answers = True
    cfg.overrefusal_guard.enabled = False
    res = certify_answer("Q", "Unverifiable claim.", evidence=[], config=cfg)
    assert res.recommended_action in {"abstain", "refuse"}


def test_warnings_only_for_qualification_action():
    cfg = FactualityCertificationConfig(mode="gated")
    cfg.certification.allow_partial_answers = False
    cfg.certification.allow_uncertainty_qualified_answers = False
    cfg.overrefusal_guard.enabled = False
    abstain_res = certify_answer("Q", "Unverifiable claim.", evidence=[], config=cfg)
    assert "qualification_recommended" not in abstain_res.warnings

    qual_cfg = FactualityCertificationConfig(mode="shadow")
    qual_res = certify_answer("Q", "Unverifiable claim.", evidence=[], config=qual_cfg)
    if qual_res.recommended_action == "answer_with_qualifications":
        assert "qualification_recommended" in qual_res.warnings


def test_extract_atomic_claims_respects_max_claims_for_mapping():
    claims, mapping = extract_atomic_claims("A, B, C, D.", max_claims=2, split_compound_claims=True)
    assert len(claims) == 2
    total_mapped_clauses = sum(len(v) for v in mapping.values())
    assert total_mapped_clauses >= len(claims)
    assert len(mapping) == 1


def test_contradicted_heavy_routes_to_abstain_or_refuse_not_partial_or_qualification():
    ev = [
        EvidenceItem(
            id="e1",
            text="Paris is not the capital of France.",
            quality_score=1.0,
            retrieval_score=1.0,
        )
    ]
    cfg = FactualityCertificationConfig(mode="gated")
    cfg.overrefusal_guard.enabled = False
    res = certify_answer("Q", "Paris is the capital of France.", evidence=ev, config=cfg)
    assert res.recommended_action in {"abstain", "refuse"}


def test_mapping_contains_only_emitted_cleaned_clauses():
    claims, mapping = extract_atomic_claims(
        "I think, A and in my opinion, B and C.", max_claims=2, split_compound_claims=True
    )
    emitted = [c.text for c in claims]
    mapped = [x for vals in mapping.values() for x in vals]
    assert emitted == mapped


def test_overall_label_contradiction_heavy_in_gated_mode():
    ev = [
        EvidenceItem(
            id="e1",
            text="Paris is not the capital of France.",
            quality_score=1.0,
            retrieval_score=1.0,
        )
    ]
    cfg = FactualityCertificationConfig(mode="gated")
    cfg.overrefusal_guard.enabled = False
    res = certify_answer("Q", "Paris is the capital of France.", evidence=ev, config=cfg)
    assert res.overall_label in {"should_abstain", "should_refuse"}


def test_non_gated_refuse_not_forced_to_qualification_without_guard_or_scope():
    cfg = FactualityCertificationConfig(mode="shadow")
    cfg.overrefusal_guard.enabled = False
    cfg.certification.allow_partial_answers = False
    cfg.certification.allow_uncertainty_qualified_answers = True
    res = certify_answer("Q", "Unknown claim.", evidence=[], config=cfg)
    assert res.recommended_action in {"refuse", "abstain"}
