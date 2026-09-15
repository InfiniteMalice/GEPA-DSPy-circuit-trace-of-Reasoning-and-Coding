"""V5 identity stays orthogonal to rewards, overlays and evaluation conditions."""

import json
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
from rg_tracer.schema_v3.examples import V3_SYNTHETIC_EXAMPLES, V5_EVALUATION_EXAMPLES


def _case(**kwargs):
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
    plain = _case(**overrides)
    variant = _case(**overrides, stripe="PARAPHRASE", stripe_subtype="CROSS_LANGUAGE", repeat_id=2)
    assert plain.case_id == variant.case_id == case_id
    assert variant.base_case_name == variant.canonical_case_key == get_case_key(case_id)
    assert plain.reward_components == variant.reward_components
    assert variant.to_dict()["repeat_id"] == 2


def test_roundtrip_old_names_and_nested_overlays():
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
    payload = _case().to_dict()
    payload["canonical_case_id"] = 6
    with pytest.raises(ValueError, match="contradicts"):
        CaseV3Result.from_dict(payload)
    payload = _case().to_dict()
    payload["contract_provenance"]["upstream_commit"] = "different"
    with pytest.raises(ValueError, match="provenance"):
        CaseV3Result.from_dict(payload)


def test_boolean_aliases_and_mutated_canonical_flags_are_rejected():
    payload = _case().to_dict()
    payload["canonical_case_id"] = True
    with pytest.raises(ValueError, match="contradicts"):
        CaseV3Result.from_dict(payload)
    result = _case(expected_answer=None)
    result.canonical = True
    with pytest.raises(ValueError, match="contradicts"):
        result.to_dict()


def test_reports_keep_fallback_outside_case_and_coordinate_counts():
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
    with pytest.raises(ValueError):
        summarize_cases([record])


def test_perturbation_adapter_and_synthetic_metadata():
    assert stripe_for_mechanism("translation") == ("PARAPHRASE", "CROSS_LANGUAGE")
    assert stripe_for_mechanism("tool_execution_failure") == ("TOOL_ERROR", None)
    with pytest.raises(ValueError):
        stripe_for_mechanism("causal_reasoning")
    for example in V3_SYNTHETIC_EXAMPLES:
        assert example["canonical_case_key"] == get_case_key(example["case_id"])
        assert example["generator_provenance"]["generation_mode"] == "hand_authored"
        assert "required_reasoning_units" in example["ontology_targets"]


def test_self_play_writes_v5_identity_provenance_and_fallback_summary(tmp_path):
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
    from rg_tracer.runners.self_play import run_self_play

    problem = tmp_path / "problem.jsonl"
    problem.write_text(json.dumps({"id": "invalid", "answer": 3, **fields}) + "\n")
    outputs = tmp_path / "outputs"
    with pytest.raises(ValueError, match=error):
        run_self_play(problem, k=1, output_dir=outputs)
    assert not outputs.exists()
