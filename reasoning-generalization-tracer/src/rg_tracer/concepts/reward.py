"""Concept reward computation integrating semantic hooks."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from .schema import ConceptSpec

DEFAULT_WEIGHTS = {
    "match": 0.4,
    "selectivity": 0.3,
    "parsimony": 0.2,
    "transfer": 0.1,
}


def _validate_float_param(
    value: Any,
    name: str,
    min_val: float | None = None,
    max_val: float | None = None,
    *,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> float:
    "Return ``value`` as float while enforcing finite bounds."

    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - validation helper
        raise ValueError(f"{name}: finite float required") from exc
    if not math.isfinite(float_value):
        raise ValueError(f"{name}: finite float required")
    if min_val is not None:
        if min_inclusive:
            if float_value < min_val:
                raise ValueError(f"{name} >= {min_val} required")
        else:
            if float_value <= min_val:
                raise ValueError(f"{name} > {min_val} required")
    if max_val is not None:
        if max_inclusive:
            if float_value > max_val:
                raise ValueError(f"{name} <= {max_val} required")
        else:
            if float_value >= max_val:
                raise ValueError(f"{name} < {max_val} required")
    return float_value


def _filter_features(
    features: Iterable[Mapping[str, Any]],
    entailed_ids: set[str],
    contradictory_ids: set[str],
) -> list[Mapping[str, Any]]:
    filtered = []
    for feature in features:
        feature_id = str(feature.get("id", ""))
        if entailed_ids and feature_id not in entailed_ids:
            continue
        if feature_id in contradictory_ids:
            continue
        filtered.append(feature)
    return filtered


def _compute_match(
    features: Iterable[Mapping[str, Any]],
    concept: ConceptSpec,
) -> float:
    expected = set(concept.expected_substructures)
    if not expected:
        return 1.0
    tags = {tag for feature in features for tag in feature.get("tags", [])}
    if not tags:
        return 0.0
    return len(tags & expected) / len(expected)


def _compute_selectivity(
    features: Iterable[Mapping[str, Any]],
    target_tags: set[str],
) -> float:
    if not target_tags:
        return 1.0
    importance_target = 0.0
    importance_other = 0.0
    for feature in features:
        weight = float(feature.get("importance", 0.0) or 0.0)
        if set(feature.get("tags", [])) & target_tags:
            importance_target += weight
        else:
            importance_other += weight
    total = importance_target + importance_other
    if total == 0:
        return 0.0
    return importance_target / total


def _compute_parsimony(
    trace_json: Mapping[str, Any],
    target_tags: set[str],
    features: Iterable[Mapping[str, Any]],
) -> float:
    lengths = trace_json.get("path_lengths", {})
    mean_length = float(lengths.get("mean", 0.0) or 0.0)
    if mean_length <= 0:
        return 0.0
    target_ids = {str(feature.get("id", "")) for feature in features}
    target_weight = sum(
        float(edge.get("weight", 0.0) or 0.0)
        for edge in trace_json.get("edges", [])
        if edge.get("src") in target_ids or edge.get("dst") in target_ids
    )
    return min(1.0, (target_weight + 1.0) / (mean_length + 1.0))


def _compute_transfer(task_metrics: Mapping[str, Any]) -> float:
    reuse = float(task_metrics.get("concept_reuse", 0.0) or 0.0)
    support = float(task_metrics.get("supporting_tasks", 1.0) or 1.0)
    if support <= 0:
        return 0.0
    return min(1.0, reuse / support)


def compute_concept_reward(
    trace_json: Mapping[str, Any],
    concept_spec: ConceptSpec,
    task_metrics: Mapping[str, Any] | None = None,
    *,
    weights: Mapping[str, float] | None = None,
    alignment: float | None = None,
    alignment_scale: float = 0.25,
) -> float:
    "Aggregate core concept scores additively, then scale the sum by alignment."
    if task_metrics is None:
        task_metrics = {}
    if weights is None:
        weights = DEFAULT_WEIGHTS
    raw_entailed = task_metrics.get("entailed_feature_ids") if task_metrics else []
    raw_contradictory = task_metrics.get("contradictory_feature_ids") if task_metrics else []
    entailed_iter = raw_entailed or []
    contradictory_iter = raw_contradictory or []
    entailed_ids = {str(value) for value in entailed_iter}
    contradictory_ids = {str(value) for value in contradictory_iter}
    features = _filter_features(
        trace_json.get("features", []),
        entailed_ids,
        contradictory_ids,
    )
    target_tags = set(concept_spec.expected_substructures)
    match = _compute_match(features, concept_spec)
    selectivity = _compute_selectivity(features, target_tags)
    parsimony = _compute_parsimony(trace_json, target_tags, features)
    transfer = _compute_transfer(task_metrics)
    reward = (
        weights.get("match", 0.0) * match
        + weights.get("selectivity", 0.0) * selectivity
        + weights.get("parsimony", 0.0) * parsimony
        + weights.get("transfer", 0.0) * transfer
    )
    if contradictory_ids:
        penalty = min(1.0, len(contradictory_ids) / (len(features) + 1.0))
        reward -= 0.2 * penalty
    if alignment is not None:
        # Allow alignment to modulate reward asymmetrically: the admissible range
        # is open so sentinel extremes are rejected, while the scale range is
        # closed on zero to let configurations disable modulation. Negative
        # alignment is neutralised so the reward is never penalised by concept
        # disagreement; callers should encode penalties elsewhere if desired.
        # Positive alignment is clamped to 1.0 to avoid runaway multipliers.
        alignment_value = _validate_float_param(
            alignment,
            "alignment",
            -1_000_000.0,
            1_000_000.0,
            min_inclusive=False,
            max_inclusive=False,
        )
        scale_value = _validate_float_param(
            alignment_scale,
            "alignment_scale",
            0.0,
            10.0,
        )
        positive_alignment = min(1.0, max(0.0, alignment_value))
        multiplier = 1.0 + scale_value * positive_alignment
        reward *= multiplier
    return float(max(0.0, reward))


__all__ = ["compute_concept_reward", "DEFAULT_WEIGHTS"]
