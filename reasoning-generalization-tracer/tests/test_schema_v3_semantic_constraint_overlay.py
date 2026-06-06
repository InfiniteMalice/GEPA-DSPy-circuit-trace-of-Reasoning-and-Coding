"""Schema V3 tests for the semantic constraint overlay."""

from __future__ import annotations

from rg_tracer.schema_v3.case_v3 import SemanticConstraintOverlay, classify_case_v3


def test_case_v3_serializes_semantic_constraint_overlay_without_changing_case_ids():
    result = classify_case_v3(
        output_text="4",
        expected_answer="4",
        is_idk=False,
        confidence=0.9,
        thought_aligned=True,
        semantic_constraint_overlay=SemanticConstraintOverlay(
            mode="shadow",
            compiled_constraint_count=2,
            verified_constraint_count=2,
            safe_for_shadow_projection=True,
        ),
    )
    payload = result.to_dict()
    assert result.case_id in set(range(18))
    assert payload["semantic_constraint_overlay"]["compiled_constraint_count"] == 2
