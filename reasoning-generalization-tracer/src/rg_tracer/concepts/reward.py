"""Concept reward computation integrating semantic hooks."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .schema import ConceptSpec

DEFAULT_WEIGHTS = {
    "match": 0.4,
    "selectivity": 0.3,
    "parsimony": 0.2,
    "transfer": 0.1,
}


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
    """Return concept reward combining match, selectivity, parsimony, transfer, and alignment."""
    if task_metrics is None:
        task_metrics = {}
    if weights is None:
        weights = DEFAULT_WEIGHTS
    entailed_ids = {str(value) for value in task_metrics.get("entailed_feature_ids", [])}
    contradictory_ids = {str(value) for value in task_metrics.get("contradictory_feature_ids", [])}
    features = _filter_features(trace_json.get("features", []), entailed_ids, contradictory_ids)
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
        reward *= 1.0 + alignment_scale * max(0.0, float(alignment))
    return float(max(0.0, reward))


__all__ = ["compute_concept_reward", "DEFAULT_WEIGHTS"]
