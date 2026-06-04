"""Schema V3 tests for the lattice deduction overlay."""

from __future__ import annotations

from rg_tracer.schema_v3.case_v3 import LatticeDeductionOverlay, classify_case_v3


def test_case_v3_serializes_lattice_overlay_without_changing_case_ids():
    result = classify_case_v3(
        output_text="5",
        expected_answer="5",
        is_idk=False,
        confidence=0.9,
        thought_aligned=True,
        lattice_deduction_overlay=LatticeDeductionOverlay(
            mode="gated",
            adapter_name="addition",
            resolved=True,
            projection_count=1,
        ),
    )
    payload = result.to_dict()
    assert result.case_id in set(range(18))
    assert payload["lattice_deduction_overlay"]["adapter_name"] == "addition"
