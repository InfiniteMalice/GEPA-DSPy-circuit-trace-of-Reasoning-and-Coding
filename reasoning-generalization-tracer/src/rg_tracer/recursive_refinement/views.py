"""MDT-inspired public view routing for toy recursive refinement."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from .types import RefinementState, ViewOperation


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def _copy_with_operation(
    state: RefinementState,
    *,
    context: Mapping[str, object],
    view_name: str,
    operation_name: str,
    rationale: str,
    prediction: object | None,
    confidence_delta: float,
    uncertainty_delta: float,
    constraint_passed: bool | None = None,
) -> RefinementState:
    changed_prediction = prediction != state.prediction
    new_confidence = _clamp_probability(state.confidence + confidence_delta)
    new_uncertainty = _clamp_probability(state.uncertainty + uncertainty_delta)
    operation = ViewOperation(
        view_name=view_name,
        operation_name=operation_name,
        step_index=len(state.operations),
        rationale=rationale,
        revisited=view_name in state.active_views,
        changed_prediction=changed_prediction,
        reduced_uncertainty=new_uncertainty < state.uncertainty,
        increased_confidence=new_confidence > state.confidence,
        constraint_passed=constraint_passed,
    )
    active_views = [*state.active_views, view_name]
    return replace(
        state,
        depth=state.depth + 1,
        state_value=new_confidence - new_uncertainty,
        uncertainty=new_uncertainty,
        confidence=new_confidence,
        prediction=prediction,
        active_views=active_views,
        operations=[*state.operations, operation],
    )


def _numbers(context: Mapping[str, object]) -> list[int]:
    raw = context.get("numbers", [])
    if not isinstance(raw, (list, tuple)):
        return []
    values = []
    for item in raw:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            values.append(item)
    return values


def _sequence(context: Mapping[str, object]) -> list[int]:
    raw = context.get("sequence", [])
    if not isinstance(raw, (list, tuple)):
        return []
    values = []
    for item in raw:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            values.append(item)
    return values


def _expected_prediction(context: Mapping[str, object]) -> object | None:
    task = context.get("task")
    numbers = _numbers(context)
    sequence = _sequence(context)
    if numbers:
        return sum(numbers)
    if task == "parity" and sequence:
        return sum(sequence) % 2
    return None


def _prediction_as_int(prediction: object | None) -> int | None:
    if isinstance(prediction, bool) or not isinstance(prediction, int):
        return None
    return prediction


@dataclass(frozen=True)
class ReasoningView:
    """A named toy refinement view with an applicability predicate."""

    name: str
    description: str
    apply: Callable[[RefinementState, Mapping[str, object]], RefinementState]
    applicable: Callable[[Mapping[str, object]], bool]


def _arithmetic_apply(
    state: RefinementState,
    context: Mapping[str, object],
) -> RefinementState:
    prediction = _expected_prediction(context)
    if prediction is None:
        prediction = state.prediction
    return _copy_with_operation(
        state,
        context=context,
        view_name="arithmetic",
        operation_name="sum_numbers",
        rationale="Compute the toy numeric invariant from public inputs.",
        prediction=prediction,
        confidence_delta=0.20,
        uncertainty_delta=-0.16,
        constraint_passed=prediction is not None,
    )


def _parity_apply(
    state: RefinementState,
    context: Mapping[str, object],
) -> RefinementState:
    task = context.get("task")
    prediction = state.prediction
    constraint_passed = None
    if task == "parity":
        sequence = _sequence(context)
        if sequence:
            prediction = sum(sequence) % 2
            constraint_passed = True
    else:
        expected = _expected_prediction(context)
        pred_int = _prediction_as_int(prediction)
        if isinstance(expected, int) and pred_int is not None:
            constraint_passed = (pred_int % 2) == (expected % 2)
    confidence_delta = 0.08 if constraint_passed is not False else -0.08
    uncertainty_delta = -0.06 if constraint_passed is not False else 0.08
    return _copy_with_operation(
        state,
        context=context,
        view_name="parity",
        operation_name="parity_invariant",
        rationale="Check or compute the parity-facing public invariant.",
        prediction=prediction,
        confidence_delta=confidence_delta,
        uncertainty_delta=uncertainty_delta,
        constraint_passed=constraint_passed,
    )


def _constraint_apply(
    state: RefinementState,
    context: Mapping[str, object],
) -> RefinementState:
    expected = _expected_prediction(context)
    constraint_passed = expected is None or state.prediction == expected
    prediction = state.prediction if constraint_passed else expected
    return _copy_with_operation(
        state,
        context=context,
        view_name="constraint_check",
        operation_name="public_constraint_check",
        rationale="Compare the prediction against explicit toy-task constraints.",
        prediction=prediction,
        confidence_delta=0.12 if constraint_passed else -0.03,
        uncertainty_delta=-0.10 if constraint_passed else -0.02,
        constraint_passed=constraint_passed,
    )


def _counterexample_apply(
    state: RefinementState,
    context: Mapping[str, object],
) -> RefinementState:
    expected = _expected_prediction(context)
    contradiction = expected is not None and state.prediction != expected
    prediction = expected if contradiction else state.prediction
    return _copy_with_operation(
        state,
        context=context,
        view_name="counterexample_check",
        operation_name="toy_counterexample_probe",
        rationale="Probe for a simple public counterexample to the current prediction.",
        prediction=prediction,
        confidence_delta=-0.04 if contradiction else 0.10,
        uncertainty_delta=0.04 if contradiction else -0.08,
        constraint_passed=not contradiction,
    )


def _verification_apply(
    state: RefinementState,
    context: Mapping[str, object],
) -> RefinementState:
    answer = context.get("answer")
    verified = answer is None or state.prediction == answer
    return _copy_with_operation(
        state,
        context=context,
        view_name="verification",
        operation_name="answer_verification",
        rationale="Verify the surfaced prediction against available public labels.",
        prediction=state.prediction,
        confidence_delta=0.16 if verified else -0.18,
        uncertainty_delta=-0.14 if verified else 0.15,
        constraint_passed=verified,
    )


def _compression_apply(
    state: RefinementState,
    context: Mapping[str, object],
) -> RefinementState:
    return _copy_with_operation(
        state,
        context=context,
        view_name="compression",
        operation_name="route_compression",
        rationale="Compress repeated public route evidence into a compact candidate.",
        prediction=state.prediction,
        confidence_delta=0.04,
        uncertainty_delta=-0.04,
        constraint_passed=None,
    )


def _has_numeric_task(context: Mapping[str, object]) -> bool:
    return bool(_numbers(context) or _sequence(context))


def _parity_applicable(context: Mapping[str, object]) -> bool:
    return (
        context.get("task") == "parity"
        or context.get("concept") == "parity"
        or _has_numeric_task(context)
    )


def build_default_view_registry() -> dict[str, ReasoningView]:
    """Return the default toy-task view registry."""

    return {
        "arithmetic": ReasoningView(
            name="arithmetic",
            description="Toy arithmetic refinement over public numeric fields.",
            apply=_arithmetic_apply,
            applicable=lambda context: bool(_numbers(context)),
        ),
        "parity": ReasoningView(
            name="parity",
            description="Toy parity invariant check or parity prediction.",
            apply=_parity_apply,
            applicable=_parity_applicable,
        ),
        "constraint_check": ReasoningView(
            name="constraint_check",
            description="Public constraint and invariant checking.",
            apply=_constraint_apply,
            applicable=_has_numeric_task,
        ),
        "counterexample_check": ReasoningView(
            name="counterexample_check",
            description="Simple public counterexample probing for toy tasks.",
            apply=_counterexample_apply,
            applicable=_has_numeric_task,
        ),
        "verification": ReasoningView(
            name="verification",
            description="Public answer verification where labels are available.",
            apply=_verification_apply,
            applicable=lambda context: "answer" in context,
        ),
        "compression": ReasoningView(
            name="compression",
            description="Compact route summarization for repeated view evidence.",
            apply=_compression_apply,
            applicable=lambda context: True,
        ),
    }


__all__ = ["ReasoningView", "build_default_view_registry"]
