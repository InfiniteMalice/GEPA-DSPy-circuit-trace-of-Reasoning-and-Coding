"""V5 identity stays orthogonal to rewards, overlays and evaluation conditions."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from rg_tracer.epistemic_cases import (
    contract_provenance,
    get_case_key,
    legacy_case_aliases,
    resolve_legacy_case_name,
)
from rg_tracer.epistemic_cases.perturbations import stripe_for_mechanism
from rg_tracer.epistemic_cases.reporting import summarize_cases
from rg_tracer.schema_v3 import classify_case_v3
from rg_tracer.schema_v3.case_v3 import CaseV3Result, ObservabilityOverlay
from rg_tracer.schema_v3.case_v3 import (
    ControlOverlay,
    GroupTheoreticOverlay,
    MDLControlOverlay,
    ReasoningOverlay,
)
from rg_tracer.schema_v3.examples import V3_SYNTHETIC_EXAMPLES, V5_EVALUATION_EXAMPLES


def _case(**kwargs):
    """Classify a correct confident answer with per-test input overrides."""
    values = dict(
        output_text="5", expected_answer="5", is_idk=False, confidence=0.9, thought_aligned=True
    )
    values.update(kwargs)
    return classify_case_v3(**values)


@pytest.mark.parametrize(
    "case_id,overrides",
    [
        (1, {}),
        (2, {"thought_aligned": False}),
        (3, {"confidence": 0.2}),
        (4, {"confidence": 0.2, "thought_aligned": False}),
        (5, {"output_text": "7"}),
        (6, {"output_text": "7", "thought_aligned": False}),
        (7, {"output_text": "7", "confidence": 0.2}),
        (8, {"output_text": "7", "confidence": 0.2, "thought_aligned": False}),
        (9, {"is_idk": True, "hidden_answer_supported": True}),
        (10, {"is_idk": True}),
        (11, {"is_idk": True, "thought_aligned": False}),
        (12, {"is_idk": True, "confidence": 0.2}),
        (13, {"is_idk": True, "confidence": 0.2, "thought_aligned": False}),
        (
            14,
            {
                "ambiguity_mode": "clarify",
                "ambiguity_high_stakes": True,
                "targeted_clarification": True,
            },
        ),
        (15, {"ambiguity_mode": "answer", "ambiguity_high_stakes": True}),
        (16, {"ambiguity_mode": "clarify", "ambiguity_high_stakes": False}),
        (
            17,
            {
                "ambiguity_mode": "clarify",
                "ambiguity_high_stakes": True,
                "stalled_after_clarification": True,
            },
        ),
    ],
)
def test_all_classifier_outcomes_across_stripes(case_id, overrides):
    """All 17 cases preserve rewards when only stripe and repeat coordinates change."""
    plain = _case(**overrides)
    variant = _case(**overrides, stripe="PARAPHRASE", stripe_subtype="CROSS_LANGUAGE", repeat_id=2)
    assert plain.case_id == variant.case_id == case_id
    assert variant.base_case_name == variant.canonical_case_key == get_case_key(case_id)
    assert plain.reward_components == variant.reward_components
    assert variant.to_dict()["repeat_id"] == 2


def test_roundtrip_old_names_and_nested_overlays():
    """Round trips preserve overlays while legacy names normalize to canonical keys."""
    for old, new in legacy_case_aliases().items():
        assert resolve_legacy_case_name(old) == new
    result = _case(observability=ObservabilityOverlay(tier="O3", has_provenance=True))
    payload = result.to_dict()
    assert CaseV3Result.from_dict(json.loads(json.dumps(payload))).to_dict() == payload
    for name in (
        "framework_version",
        "canonical_case_id",
        "canonical_case_key",
        "canonical_case_title",
        "canonical",
        "contract_provenance",
        "overlay_version",
    ):
        payload.pop(name)
    payload["base_case_name"] = "confident_correct_aligned_answer"
    restored = CaseV3Result.from_dict(payload)
    assert restored.base_case_name == get_case_key(1)
    assert restored.observability == result.observability


def test_conflicting_serialized_identity_or_provenance_is_rejected():
    """Deserialization rejects contradictory canonical IDs and upstream provenance."""
    payload = _case().to_dict()
    payload["canonical_case_id"] = 6
    with pytest.raises(ValueError, match="contradicts"):
        CaseV3Result.from_dict(payload)
    payload = _case().to_dict()
    payload["contract_provenance"]["upstream_commit"] = "different"
    with pytest.raises(ValueError, match="provenance"):
        CaseV3Result.from_dict(payload)


def test_boolean_aliases_and_mutated_canonical_flags_are_rejected():
    """Identity validation rejects boolean ID aliases and mutated canonical flags."""
    payload = _case().to_dict()
    payload["canonical_case_id"] = True
    with pytest.raises(ValueError, match="contradicts"):
        CaseV3Result.from_dict(payload)
    result = _case(expected_answer=None)
    result.canonical = True
    with pytest.raises(ValueError, match="contradicts"):
        result.to_dict()


def test_reports_keep_fallback_outside_case_and_coordinate_counts():
    """Reports count fallback separately from canonical cases and evaluation slices."""
    report = summarize_cases(V5_EVALUATION_EXAMPLES + [{"case_id": 0}, {"reward_case": 0}])
    assert tuple(report["case_counts"]) == tuple(range(1, 18))
    assert report["canonical_count"] == 6
    assert report["unclassified_count"] == 2
    assert len(report["coordinate_counts"]) == 6
    assert 0 not in report["case_counts"]


@pytest.mark.parametrize(
    "record",
    [
        {"case_id": 1, "canonical_case_id": 6},
        {"case_id": 0, "canonical": True},
        {"case_id": 1, "framework_version": "17case-v6"},
    ],
)
def test_reports_reject_conflicting_identity(record):
    """Summary validation rejects contradictory IDs, flags and framework versions."""
    with pytest.raises(ValueError):
        summarize_cases([record])


@pytest.mark.parametrize(
    "record",
    [
        {"case_id": 1, "reward_case": 6},
        {"canonical_case_id": 1, "reward_case": 6},
        {"case_id": 0, "reward_case": 1},
        {"case_id": 1, "reward_case": True},
        {"case_id": 1, "reward_case": 1.0},
        {"case_id": 1, "canonical_case_key": "incorrect"},
        {"case_id": 1, "canonical_case_title": "incorrect"},
        {"case_id": 0, "canonical_case_key": "incorrect"},
        {"case_id": 0, "canonical_case_title": "incorrect"},
        {"case_id": 0, "canonical_case_id": 0},
        {"case_id": 1, "canonical_case_id": None},
        {"case_id": 1, "canonical_case_key": None},
        {"case_id": 1, "canonical_case_title": None},
    ],
)
def test_reports_reject_all_supplied_identity_conflicts(record):
    """Legacy aliases and supplied canonical labels cannot silently contradict IDs."""
    with pytest.raises(ValueError):
        summarize_cases([record])


@pytest.mark.parametrize(
    "record,canonical,fallback",
    [
        ({}, 0, 1),
        ({"case_id": None, "reward_case": 1}, 1, 0),
        ({"canonical_case_id": None, "reward_case": 0}, 0, 1),
        ({"case_id": 1, "reward_case": None}, 1, 0),
        ({"case_id": 0, "canonical_case_key": None, "canonical_case_title": None}, 0, 1),
    ],
)
def test_reports_preserve_missing_and_nullable_ids(record, canonical, fallback):
    """Missing legacy IDs allow another nonnull ID or the operational fallback."""
    report = summarize_cases([record])
    assert report["canonical_count"] == canonical
    assert report["unclassified_count"] == fallback


@pytest.mark.parametrize("field", ["case_id", "canonical_case_id", "reward_case"])
@pytest.mark.parametrize("value", [True, False, 1.0, 0.0, "1", -1, 18])
def test_reports_reject_noncanonical_id_types_and_ranges(field, value):
    """Every ID source rejects noninteger aliases and values outside Cases 0–17."""
    with pytest.raises(ValueError):
        summarize_cases([{field: value}])


def test_reports_accept_matching_canonical_labels_and_legacy_id():
    """Complete serialized identity and a matching legacy ID remain countable."""
    payload = _case().to_dict()
    payload["reward_case"] = 1
    report = summarize_cases([payload])
    assert report["case_counts"][1] == 1
    assert report["unclassified_count"] == 0


def _reward_overlays():
    """Provide nonzero contributions from each diagnostic reward component."""
    return {
        "observability": ObservabilityOverlay(tier="O5", has_provenance=True),
        "control_overlay": ControlOverlay(required_controls=["check"], observed_controls=["check"]),
        "reasoning_overlay": ReasoningOverlay(required_units=["sum"], observed_units=["sum"]),
        "group_theoretic_overlay": GroupTheoreticOverlay(invariant_properties=["parity"]),
        "mdl_control_overlay": MDLControlOverlay(escalation_required=True, escalation_taken=True),
    }


@pytest.mark.parametrize("is_idk", [False, True])
@pytest.mark.parametrize("output_text", ["5", "7"])
def test_missing_confidence_fallback_has_neutral_rewards(is_idk, output_text):
    """Unknown-confidence fallback erases scoring surrogates and diagnostic bonuses."""
    result = _case(confidence=None, is_idk=is_idk, output_text=output_text, **_reward_overlays())
    assert result.case_id == 0
    assert set(asdict(result.reward_components).values()) == {0.0}


@pytest.mark.parametrize(
    "mode,high_stakes,flags,case_id",
    [
        ("clarify", True, {"targeted_clarification": True}, 14),
        ("answer", True, {}, 15),
        ("clarify", False, {}, 16),
        ("clarify", True, {"stalled_after_clarification": True}, 17),
    ],
)
def test_missing_confidence_ambiguity_rewards_remain_unchanged(mode, high_stakes, flags, case_id):
    """Ambiguity rewards retain the existing low-confidence computation and overlays."""
    kwargs = dict(ambiguity_mode=mode, ambiguity_high_stakes=high_stakes, **flags)
    unknown = _case(confidence=None, **kwargs, **_reward_overlays())
    observed = _case(confidence=0.0, **kwargs, **_reward_overlays())
    assert unknown.case_id == observed.case_id == case_id
    assert unknown.reward_components == observed.reward_components
    assert unknown.reward_components.r_grounding == 0.25
    assert unknown.reward_components.r_observability == 0.4


@pytest.mark.parametrize(
    "confidence,is_idk,case_id,token,confidence_reward,abstain,total",
    [
        (0.9, False, 1, 2.0, 0.0, 0.0, 4.2),
        (0.9, True, 10, 0.0, -2.0, 0.0, 0.2),
        (0.2, False, 3, 1.0, 0.0, 0.0, 3.2),
        (0.2, True, 12, 0.0, 0.0, 0.25, 2.45),
    ],
)
def test_observed_confidence_preserves_historical_rewards(
    confidence, is_idk, case_id, token, confidence_reward, abstain, total
):
    """Observed high/low answer and IDK rewards retain pre-fix values plus overlays."""
    result = _case(confidence=confidence, is_idk=is_idk, **_reward_overlays())
    assert result.case_id == case_id
    assert asdict(result.reward_components) == pytest.approx(
        dict(
            r_token=token,
            r_confidence=confidence_reward,
            r_thought=1.0,
            r_abstain=abstain,
            r_grounding=0.25,
            r_control=0.35,
            r_reasoning_unit=0.1,
            r_observability=0.4,
            r_group_theoretic=0.1,
            total=total,
        )
    )


def test_perturbation_adapter_and_synthetic_metadata():
    """Perturbations resolve to stripes and synthetic examples retain ontology metadata."""
    assert stripe_for_mechanism("translation") == ("PARAPHRASE", "CROSS_LANGUAGE")
    assert stripe_for_mechanism("tool_execution_failure") == ("TOOL_ERROR", None)
    with pytest.raises(ValueError):
        stripe_for_mechanism("causal_reasoning")
    for example in V3_SYNTHETIC_EXAMPLES:
        assert example["canonical_case_key"] == get_case_key(example["case_id"])
        assert example["generator_provenance"]["generation_mode"] == "hand_authored"
        assert "required_reasoning_units" in example["ontology_targets"]


def test_self_play_writes_v5_identity_provenance_and_fallback_summary(tmp_path):
    """Self-play persists V5 provenance and counts missing-answer fallback separately."""
    from rg_tracer.runners.self_play import run_self_play

    problem_path = tmp_path / "problem.jsonl"
    problem_path.write_text(
        json.dumps(
            {
                "id": "v5-fallback",
                "numbers": [1, 2],
                "stripe": "MISSING_EVIDENCE",
                "repeat_id": 3,
            }
        )
        + "\n"
    )
    result = run_self_play(problem_path, k=1, output_dir=tmp_path)
    tmp_path = Path(result["run_dir"])
    records = [json.loads(line) for line in (tmp_path / "scores.jsonl").read_text().splitlines()]
    assert result
    assert records[0]["case_id"] == 0
    assert records[0]["canonical"] is False
    assert records[0]["repeat_id"] == 3
    assert records[0]["stripe"] == "MISSING_EVIDENCE"
    assert records[0]["contract_provenance"] == contract_provenance()
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["ontology_version"] == "rg-ontology-v1"
    assert metadata["overlay_version"] == "v3"
    report = json.loads((tmp_path / "case_summary.json").read_text())
    assert report["unclassified_count"] == 1
    assert report["canonical_count"] == 0


@pytest.mark.parametrize("targets", [None, {"required_concepts": ["parity"]}])
def test_self_play_optional_ontology_targets(tmp_path, targets):
    """Self-play preserves supplied ontology targets and normalizes missing targets."""
    from rg_tracer.runners.self_play import run_self_play

    problem = tmp_path / "problem.jsonl"
    problem.write_text(
        json.dumps(
            {
                "id": "ontology-annotations",
                "numbers": [1, 2],
                "answer": 3,
                "ontology_targets": targets,
            }
        )
        + "\n"
    )
    result = run_self_play(problem, k=1, output_dir=tmp_path)
    record = json.loads((Path(result["run_dir"]) / "scores.jsonl").read_text().strip())
    assert record["ontology_targets"] == (targets or {})


@pytest.mark.parametrize(
    "fields,error",
    [
        ({"ontology_targets": "invalid"}, "ontology_targets"),
        ({"stripe": "INVENTED"}, "stripe"),
        ({"repeat_id": -1}, "repeat_id"),
    ],
)
def test_self_play_rejects_bad_metadata_before_creating_outputs(tmp_path, fields, error):
    """Invalid ontology targets or coordinates fail before output directories exist."""
    from rg_tracer.runners.self_play import run_self_play

    problem = tmp_path / "problem.jsonl"
    problem.write_text(json.dumps({"id": "invalid", "answer": 3, **fields}) + "\n")
    outputs = tmp_path / "outputs"
    with pytest.raises(ValueError, match=error):
        run_self_play(problem, k=1, output_dir=outputs)
    assert not outputs.exists()
