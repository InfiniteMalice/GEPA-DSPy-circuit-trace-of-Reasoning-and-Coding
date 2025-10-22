import pytest

from rg_tracer.scoring.aggregator import (
    DEFAULT_GATES,
    Profile,
    apply_hard_gates,
    evaluate_profile,
    weighted_geometric_mean,
)


def test_weighted_geometric_mean_basic():
    scores = {"a": 4, "b": 2}
    weights = {"a": 0.5, "b": 0.5}
    value = weighted_geometric_mean(scores, weights, epsilon=0.001)
    assert value == pytest.approx(2.829, abs=1e-3)


def test_apply_hard_gates_enforces_threshold():
    passes, failed = apply_hard_gates({"logical_validity": 4, "rigor": 2, "numerical_accuracy": 3})
    assert not passes
    assert failed["rigor"] == 2


def test_evaluate_profile_respects_profile_weights():
    profile = Profile(
        name="demo", weights={"logical_validity": 1.0, "rigor": 1.0, "numerical_accuracy": 1.0}
    )
    scores = {"logical_validity": 4, "rigor": 4, "numerical_accuracy": 3}
    result = evaluate_profile(scores, profile)
    assert result["passes_gates"]
    assert result["composite"] > 3.5
