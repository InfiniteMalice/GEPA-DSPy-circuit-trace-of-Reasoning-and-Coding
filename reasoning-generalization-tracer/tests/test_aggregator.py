import pytest

from rg_tracer.scoring.aggregator import (
    Profile,
    apply_hard_gates,
    evaluate_profile,
    get_last_config,
    load_profiles,
    _split_profile_payload,
    weighted_geometric_mean,
)


def test_weighted_geometric_mean_basic():
    scores = {"a": 4, "b": 2}
    weights = {"a": 0.5, "b": 0.5}
    value = weighted_geometric_mean(scores, weights, epsilon=0.001)
    assert value == pytest.approx(2.829, abs=1e-3)


def test_apply_hard_gates_enforces_threshold():
    passes, failed = apply_hard_gates(
        {
            "logical_validity": 4,
            "rigor": 2,
            "numerical_accuracy": 3,
        }
    )
    assert not passes
    assert failed["rigor"] == 2


def test_evaluate_profile_respects_profile_weights():
    profile = Profile(
        name="demo",
        weights={
            "logical_validity": 1.0,
            "rigor": 1.0,
            "numerical_accuracy": 1.0,
        },
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
    second = get_last_config()
    assert second is not config
    assert second == config
    second["attr"]["probe_size"] = 99
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


def test_fallback_parser_handles_nested_config_and_bonuses(tmp_path, monkeypatch):
    from rg_tracer.scoring import aggregator as agg

    monkeypatch.setattr(agg, "yaml", None)
    path = tmp_path / "profiles.yaml"
    yaml_text = (
        "config:\n"
        "  attr:\n"
        "    thresholds:\n"
        "      entropy:\n"
        "        min: 0.1\n"
        "profiles:\n"
        "  demo:\n"
        "    weights:\n"
        "      rigor: 1\n"
        "    bonuses:\n"
        "      alignment_gain: 0.02\n"
    )
    path.write_text(yaml_text, encoding="utf8")
    profiles = load_profiles(path)
    assert set(profiles.keys()) == {"demo"}
    config = get_last_config()
    assert config["attr"]["thresholds"]["entropy"]["min"] == 0.1
    assert profiles["demo"].bonuses["alignment_gain"] == 0.02


def test_fallback_parser_handles_empty_subsections(tmp_path, monkeypatch):
    from rg_tracer.scoring import aggregator as agg

    monkeypatch.setattr(agg, "yaml", None)
    path = tmp_path / "profiles.yaml"
    yaml_text = (
        "config:\n"
        "  attr:\n"
        "    thresholds:\n"
        "profiles:\n"
        "  demo:\n"
        "    weights:\n"
        "      rigor: 1\n"
        "    bonuses:\n"
    )
    path.write_text(yaml_text, encoding="utf8")
    profiles = load_profiles(path)
    assert profiles["demo"].bonuses == {}
    config = get_last_config()
    assert "thresholds" in config.get("attr", {})


def test_fallback_parser_raises_on_invalid_top_level(tmp_path, monkeypatch):
    from rg_tracer.scoring import aggregator as agg

    monkeypatch.setattr(agg, "yaml", None)
    path = tmp_path / "profiles.yaml"
    path.write_text("notes: unsupported\n", encoding="utf8")
    with pytest.raises(ValueError, match="Unable to parse line"):
        load_profiles(path)


def test_split_profile_payload_rejects_non_numeric_weights():
    payload = {"weights": {"rigor": "not-a-number"}}
    with pytest.raises(ValueError, match="rigor"):
        _split_profile_payload(payload)


def test_split_profile_payload_rejects_bool_weights():
    payload = {"weights": {"rigor": True}}
    with pytest.raises(TypeError, match="rigor"):
        _split_profile_payload(payload)


def test_split_profile_payload_rejects_bool_weights_legacy_shape():
    payload = {"rigor": True}
    with pytest.raises(TypeError, match="rigor"):
        _split_profile_payload(payload)


def test_split_profile_payload_rejects_non_mapping_weights_section():
    payload = {"weights": 1}
    with pytest.raises(TypeError, match="weights section must be a mapping"):
        _split_profile_payload(payload)


def test_default_config_includes_abstention_and_alignment():
    config = get_last_config()
    assert config["abstention"]["threshold"] == 0.75
    assert config["thought_alignment"]["theta_match"] == 0.8
