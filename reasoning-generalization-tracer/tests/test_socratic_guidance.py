from rg_tracer.socratic_guidance import (
    AssistedAttemptRecord,
    apply_reward_decay,
    assisted_improvement,
    teacher_dependency_metrics,
)


def test_assisted_score_improves_but_reward_is_decayed():
    record = AssistedAttemptRecord("t1", "a1", 0.2, 1.0, True, 0.3)
    assert assisted_improvement(record) == 0.8
    assert apply_reward_decay(record.post_guidance_score, True, record.reward_decay) == 0.7


def test_unassisted_success_receives_no_decay():
    assert apply_reward_decay(1.0, False, 0.5) == 1.0


def test_dependency_rate_computes_correctly():
    records = [
        AssistedAttemptRecord("t1", "a1", 0.2, 1.0, True, 0.3),
        AssistedAttemptRecord("t2", "a1", 1.0, 1.0, False, 0.0),
    ]
    metrics = teacher_dependency_metrics(records)
    assert metrics["guidance_dependency_rate"] == 0.5
