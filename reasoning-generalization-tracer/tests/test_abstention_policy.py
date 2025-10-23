from rg_tracer.abstention import apply_abstention, ABSTENTION_THRESHOLD


def test_apply_abstention_below_threshold():
    result = apply_abstention("Answer", ABSTENTION_THRESHOLD - 0.01, sem_score=3.0)
    assert result.abstained
    assert result.text == "I don't know."


def test_apply_abstention_above_threshold():
    result = apply_abstention("Answer", ABSTENTION_THRESHOLD + 0.01, sem_score=3.0)
    assert not result.abstained
    assert result.text == "Answer"


def test_apply_abstention_semantic_failure():
    result = apply_abstention("Answer", ABSTENTION_THRESHOLD + 0.1, sem_score=1.0)
    assert result.abstained
