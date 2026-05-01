from factuality_certification import EvidenceItem, FactualityCertificationConfig, certify_answer
from factuality_certification.integrations import reward_features


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


def test_contradicted_claim():
    ev = [
        EvidenceItem(
            id="e1",
            text="Paris is not the capital of France.",
            quality_score=1.0,
            retrieval_score=1.0,
        )
    ]
    res = certify_answer("Q", "Paris is the capital of France.", evidence=ev)
    assert any(s.support_label == "contradicted" for s in res.claim_support)


def test_overrefusal_guard():
    ev = [
        EvidenceItem(id="e1", text="Water boils at 100 C.", quality_score=1.0, retrieval_score=1.0)
    ]
    res = certify_answer("Q", "I cannot help with that.", evidence=ev)
    assert res.recommended_action != "refuse"


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
