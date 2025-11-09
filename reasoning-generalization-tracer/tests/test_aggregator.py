import pytest

from rg_tracer.scoring.aggregator import (
    Profile,
    apply_hard_gates,
    evaluate_profile,
    get_last_config,
    load_profiles,
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


def test_load_profiles_exposes_config(tmp_path):
    yaml_text = (
        "profiles:\n"
        "  demo:\n"
        "    weights:\n"
        "      logical_validity: 1.0\n"
        "    bonuses:\n"
        "      alignment_gain: 0.02\n"
        "config:\n"
        "  attr:\n"
        "    probe_size: 3\n"
        "    topk: 1\n"
        "    backend: null\n"
    )
    path = tmp_path / "profiles.yaml"
    path.write_text(yaml_text, encoding="utf8")
    profiles = load_profiles(path)
    profile = profiles["demo"]
    assert profile.bonuses["alignment_gain"] == 0.02
    config = get_last_config()
    assert config["attr"]["probe_size"] == 3
    config["attr"]["probe_size"] = 99
    assert get_last_config()["attr"]["probe_size"] == 3


def test_fallback_parser_handles_top_level_scalars(tmp_path, monkeypatch):
    from rg_tracer.scoring import aggregator as agg

    monkeypatch.setattr(agg, "yaml", None)
    path = tmp_path / "profiles.yaml"
    yaml_text = (
        "config:\n"
        "  alignment_scale: 0.25\n"
        "profiles:\n"
        "  demo:\n"
        "    weights:\n"
        "      rigor: 1\n"
    )
    path.write_text(yaml_text, encoding="utf8")
    profiles = load_profiles(path)
    assert "demo" in profiles
    config = get_last_config()
    assert config["alignment_scale"] == 0.25
