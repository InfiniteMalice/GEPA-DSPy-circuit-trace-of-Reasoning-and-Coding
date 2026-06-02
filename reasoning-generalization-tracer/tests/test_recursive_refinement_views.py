"""Tests for MDT-inspired toy view routing operators."""

from __future__ import annotations

from rg_tracer.recursive_refinement.types import RefinementState
from rg_tracer.recursive_refinement.views import build_default_view_registry


def _state(prediction=4):
    return RefinementState(
        trajectory_id="t",
        parent_id=None,
        depth=0,
        state_value=0.0,
        uncertainty=0.5,
        confidence=0.5,
        prediction=prediction,
        active_views=[],
        operations=[],
    )


def test_arithmetic_view_corrects_public_addition_prediction():
    view = build_default_view_registry()["arithmetic"]
    refined = view.apply(_state(), {"task": "addition", "numbers": [2, 3], "answer": 5})
    assert refined.prediction == 5
    assert refined.operations[-1].view_name == "arithmetic"
    assert refined.operations[-1].changed_prediction
    assert refined.operations[-1].reduced_uncertainty


def test_parity_view_records_invariant_without_replacing_addition_answer():
    view = build_default_view_registry()["parity"]
    refined = view.apply(_state(prediction=5), {"task": "addition", "numbers": [2, 3]})
    assert refined.prediction == 5
    assert refined.operations[-1].constraint_passed


def test_verification_view_records_failed_constraint():
    view = build_default_view_registry()["verification"]
    refined = view.apply(_state(prediction=4), {"task": "addition", "answer": 5})
    assert refined.operations[-1].constraint_passed is False
    assert refined.confidence < 0.5
