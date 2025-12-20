import copy

from rg_tracer.scoring import aggregator
from rg_tracer.abstention import evaluate_abstention_reward


BASE_TEXT = "Add 2 and 3 to get 5."


def _case(**kwargs):
    raw_config = kwargs.get("config", aggregator.DEFAULT_CONFIG)
    config = copy.deepcopy(raw_config)
    return evaluate_abstention_reward(
        expected_answer=kwargs.get("expected", 5),
        prediction=kwargs.get("prediction", 5),
        text=kwargs.get("text", BASE_TEXT),
        confidence=kwargs.get("confidence", 0.9),
        aligned=kwargs.get("aligned", True),
        abstained=kwargs.get("abstained", False),
        config=config,
    )


def test_all_reward_cases_are_accessible():
    cases = {
        1: _case(),
        2: _case(aligned=False),
        3: _case(confidence=0.6),
        4: _case(confidence=0.6, aligned=False),
        5: _case(prediction=7, aligned=True),
        6: _case(prediction=7, aligned=False),
        7: _case(prediction=7, aligned=True, confidence=0.6),
        8: _case(prediction=7, aligned=False, confidence=0.6),
        9: _case(abstained=True, aligned=False, confidence=0.9, prediction=5),
        10: _case(abstained=True, aligned=True, confidence=0.9, prediction="I don't know"),
        11: _case(abstained=True, aligned=False, confidence=0.9, prediction="I don't know"),
        12: _case(abstained=True, aligned=True, confidence=0.4, prediction="I don't know"),
        13: _case(abstained=True, aligned=False, confidence=0.4, prediction="I don't know"),
        0: _case(expected=None, prediction=None, text=""),
    }
    observed = {outcome.case_id for outcome in cases.values()}
    assert set(cases) == observed, f"expected case IDs {set(cases)}, observed {observed}"


def test_confidence_push_requires_alignment():
    aligned = _case()
    unaligned = _case(aligned=False)
    assert aligned.components["token"] > unaligned.components["token"]


def test_timid_expert_reward_positive():
    outcome = _case(confidence=0.6)
    assert outcome.case_id == 3
    assert outcome.reward > 0


def test_lucky_guesser_has_lower_reward():
    timid = _case(confidence=0.6)
    lucky = _case(confidence=0.6, aligned=False)
    assert timid.reward > lucky.reward
    assert lucky.case_id == 4


def test_miscalibrated_honest_idk_is_distinct():
    lazy = _case(abstained=True, aligned=False, confidence=0.9, prediction=5)
    miscal_aligned = _case(abstained=True, aligned=True, confidence=0.9, prediction="I don't know")
    miscal_unaligned = _case(
        abstained=True, aligned=False, confidence=0.9, prediction="I don't know"
    )
    grounded = _case(abstained=True, aligned=True, confidence=0.4, prediction="I don't know")
    cautious = _case(abstained=True, aligned=False, confidence=0.4, prediction="I don't know")
    assert lazy.case_id == 9
    assert miscal_aligned.case_id == 10
    assert miscal_unaligned.case_id == 11
    assert grounded.case_id == 12
    assert cautious.case_id == 13
    assert miscal_aligned.components["confidence"] < 0
    assert miscal_unaligned.components["confidence"] < 0
    assert grounded.components.get("abstain", 0) > 0
    assert cautious.components.get("abstain", 0) > 0


def test_confident_but_wrong_remains_aligned_rewarded_for_honesty():
    outcome = _case(prediction=7, aligned=True)
    assert outcome.case_id == 5
    assert outcome.components["thought"] >= 0
    assert outcome.reward < 0  # knowledge penalty dominates


def test_punctuation_only_prediction_returns_none():
    outcome = _case(
        prediction=None,
        text="... ??? !!!",
        abstained=True,
        aligned=False,
        confidence=0.6,
    )
    assert outcome.prediction is None
    assert outcome.case_id == 13


def test_punctuation_only_prediction_is_ignored():
    outcome = evaluate_abstention_reward(
        expected_answer="yes",
        prediction=None,
        text="... ???",
        confidence=0.4,
        aligned=False,
        abstained=True,
        config=copy.deepcopy(aggregator.DEFAULT_CONFIG),
    )
    assert outcome.prediction is None
    assert outcome.abstained
    assert outcome.case_id == 13


def test_miscalibrated_grounded_idk_receives_thought_reward():
    outcome = _case(abstained=True, aligned=True, confidence=0.9, prediction="I don't know")
    assert outcome.case_id == 10
    assert outcome.components["thought"] > 0


def test_miscalibrated_ungrounded_idk_no_thought_reward():
    outcome = _case(abstained=True, aligned=False, confidence=0.9, prediction="I don't know")
    assert outcome.case_id == 11
    assert outcome.components["thought"] == 0.0


def test_grounded_low_confidence_idk_receives_thought_reward():
    outcome = _case(abstained=True, aligned=True, confidence=0.4, prediction="I don't know")
    assert outcome.case_id == 12
    assert outcome.components["thought"] > 0


def test_ungrounded_low_confidence_idk_no_thought_reward():
    outcome = _case(abstained=True, aligned=False, confidence=0.4, prediction="I don't know")
    assert outcome.case_id == 13
    assert outcome.components["thought"] == 0.0
