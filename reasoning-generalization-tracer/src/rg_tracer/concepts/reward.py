"""Concept reward computation integrating semantic hooks."""

from __future__ import annotations

import math
from collections.abc import Iterable as IterableABC, Mapping as MappingABC
from typing import Any, Iterable, Mapping


from ..modules.grn import apply_grn
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
    """Return ``value`` as float while enforcing finite bounds."""

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


def _feature_tags(feature: Mapping[str, Any]) -> set[str]:
    raw = feature.get("tags")
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, IterableABC):
        tags: set[str] = set()
        for tag in raw:
            if tag is None or isinstance(tag, bool):
                continue
            tags.add(str(tag))
        return tags
    return set()


def _compute_match(
    features: Iterable[Mapping[str, Any]],
    concept: ConceptSpec,
) -> float:
    expected = set(concept.expected_substructures)
    if not expected:
        return 1.0
    tags = {tag for feature in features for tag in _feature_tags(feature)}
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
        feature_tags = _feature_tags(feature)
        if feature_tags & target_tags:
            importance_target += weight
        else:
            importance_other += weight
    total = importance_target + importance_other
    if total == 0:
        return 0.0
    return importance_target / total


def _compute_parsimony(
    trace_json: Mapping[str, Any],
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
    use_grn: bool = False,
    grn_eps: float = 1e-6,
) -> float:
    """Aggregate concept scores additively, then scale by alignment.

    When ``alignment`` is provided, the additive score is multiplied by
    ``1 + alignment_scale * clamp(alignment, 0, 1)``. The ``alignment`` input is
    validated against an open interval (-1e6, 1e6) and clamped to [0, 1] before
    scaling; ``alignment_scale`` is clamped to [0.0, 10.0] (default 0.25) so the
    multiplier stays bounded. ``alignment=None`` skips scaling (multiplier = 1).
    """
    if task_metrics is None:
        task_metrics = {}
    if weights is None:
        weights = DEFAULT_WEIGHTS
    raw_entailed = task_metrics.get("entailed_feature_ids") if task_metrics else None
    raw_contradictory = task_metrics.get("contradictory_feature_ids") if task_metrics else None

    def _iter_ids(raw: object) -> IterableABC[object]:
        if raw is None:
            return ()
        if isinstance(raw, MappingABC):
            return raw.keys()
        if isinstance(raw, (str, bytes)):
            return (raw,)
        if isinstance(raw, IterableABC):
            return raw
        return (raw,)

    entailed_ids = {str(value) for value in _iter_ids(raw_entailed)}
    contradictory_ids = {str(value) for value in _iter_ids(raw_contradictory)}
    features = _filter_features(
        trace_json.get("features") or [],
        entailed_ids,
        contradictory_ids,
    )
    if use_grn and features:
        importances = [
            float(feature.get("importance", 0.0) or 0.0) for feature in features
        ]
        normalised = apply_grn(importances, eps=grn_eps).tolist()
        aligned_features = []
        for feature, importance in zip(features, normalised, strict=True):
            updated = dict(feature)
            updated["importance"] = float(importance)
            aligned_features.append(updated)
        features = aligned_features
    target_tags = set(concept_spec.expected_substructures)
    match = _compute_match(features, concept_spec)
    selectivity = _compute_selectivity(features, target_tags)
    parsimony = _compute_parsimony(trace_json, features)
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
            min_inclusive=True,
            max_inclusive=True,
        )
        positive_alignment = min(1.0, max(0.0, alignment_value))
        multiplier = 1.0 + scale_value * positive_alignment
        reward *= multiplier
    return float(max(0.0, reward))


__all__ = ["compute_concept_reward", "DEFAULT_WEIGHTS"]
