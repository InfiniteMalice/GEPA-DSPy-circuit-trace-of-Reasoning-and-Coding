"""Offline regression contract for the pinned Mindfulness V5 mirror."""

import hashlib
from importlib.resources import files

import pytest
import yaml

from rg_tracer.epistemic_cases import (
    canonical_case_ids,
    contract_provenance,
    get_case,
    get_case_key,
    is_canonical_case,
    resolve_legacy_case_name,
    stripe_registry,
    validate_coordinate,
)
from rg_tracer.schema_v3 import classify_case_v3


def test_pinned_manifest_and_stripes_match_all_loaded_fields():
    """Verify that pinned manifest and stripes match all loaded fields."""
    root = files("rg_tracer.epistemic_cases")
    manifest = yaml.safe_load(root.joinpath("17_case_manifest.yaml").read_text())
    assert manifest["framework_version"] == "17case-v5"
    assert manifest["canonical_case_count"] == 17
    assert canonical_case_ids() == tuple(range(1, 18))
    assert not is_canonical_case(0)
    for row in manifest["cases"]:
        case = get_case(row["id"])
        for key in (
            "id",
            "key",
            "title",
            "expected_epistemic_behavior",
            "confidence_semantics",
            "stakes_semantics",
        ):
            assert getattr(case, key) == row[key]
    stripes = yaml.safe_load(root.joinpath("robustness_stripes.yaml").read_text())
    assert stripes["registry_version"] == "17case-v5"
    assert stripe_registry() == {row["id"]: row for row in stripes["stripes"]}
    provenance = contract_provenance()
    assert provenance["upstream_commit"] == "a12d8d27134bc38b5aa43aff33b84db2f7f51bce"
    # Independent anchors from `git show <upstream SHA>:<path>`, not the metadata file.
    assert provenance["local_manifest_hash"] == (
        "dd65edf21e951031846e1eb5717671984843d01747938a76a068d7ae256e9ffe"
    )
    assert provenance["local_stripe_registry_hash"] == (
        "3b878df066c1d0364237de5cf4ff077ce93f18a9443a62df8067429ab8773722"
    )
    for filename, key in (
        ("17_case_manifest.yaml", "local_manifest_hash"),
        ("robustness_stripes.yaml", "local_stripe_registry_hash"),
    ):
        assert provenance[key] == hashlib.sha256(root.joinpath(filename).read_bytes()).hexdigest()


@pytest.mark.parametrize("value", [True, 1.0, "1", -1, 18, None])
def test_invalid_case_ids_are_rejected(value):
    """Verify that invalid case ids are rejected."""
    assert not is_canonical_case(value)
    with pytest.raises(ValueError):
        get_case(value)
    with pytest.raises(ValueError):
        validate_coordinate(value)


@pytest.mark.parametrize(
    "stripe,subtype,repeat",
    [
        ("UNKNOWN", None, 0),
        ("NONE", "CROSS_LANGUAGE", 0),
        ("PARAPHRASE", "SEMANTIC_LAUNDERING", 0),
        ("NONE", None, -1),
        ("NONE", None, True),
        ("NONE", None, 1.5),
    ],
)
def test_invalid_evaluation_coordinates(stripe, subtype, repeat):
    """Verify that invalid evaluation coordinates are rejected."""
    with pytest.raises(ValueError):
        validate_coordinate(1, stripe, subtype, repeat)


def test_legacy_normalization_and_fallback():
    """Verify legacy-name normalization and non-canonical fallback behavior."""
    assert resolve_legacy_case_name("timid_expert_aligned_answer") == get_case_key(3)
    assert resolve_legacy_case_name(get_case_key(6)) == get_case_key(6)
    with pytest.raises(ValueError):
        resolve_legacy_case_name("invented_case")
    result = classify_case_v3(
        output_text="",
        expected_answer=None,
        is_idk=False,
        confidence=None,
        thought_aligned=False,
    ).to_dict()
    assert result["case_id"] == 0
    assert result["canonical_case_id"] is None
    assert result["canonical"] is False
    assert result["framework_version"] == "17case-v5"


def test_low_stakes_assumptive_answer_is_not_high_stakes_failure():
    """Verify that low stakes assumptive answer is not high stakes failure."""
    result = classify_case_v3(
        output_text="5",
        expected_answer="5",
        is_idk=False,
        confidence=0.9,
        thought_aligned=True,
        ambiguity_mode="assumptive_proceed",
        ambiguity_high_stakes=False,
    )
    assert result.case_id == 1


def test_repeated_targeted_questions_are_a_loop():
    """Verify that repeated targeted questions are a loop."""
    result = classify_case_v3(
        output_text="Which account?",
        expected_answer=None,
        is_idk=False,
        confidence=None,
        thought_aligned=True,
        ambiguity_mode="clarify",
        ambiguity_high_stakes=True,
        targeted_clarification=True,
        excessive_questions=True,
    )
    assert result.case_id == 17


def test_missing_confidence_does_not_claim_an_observed_low_confidence_band():
    """Verify that missing confidence does not claim an observed low confidence band."""
    result = classify_case_v3(
        output_text="5",
        expected_answer="5",
        is_idk=False,
        confidence=None,
        thought_aligned=True,
    )
    assert result.confidence_band == "unknown"
    assert result.case_id == 0
