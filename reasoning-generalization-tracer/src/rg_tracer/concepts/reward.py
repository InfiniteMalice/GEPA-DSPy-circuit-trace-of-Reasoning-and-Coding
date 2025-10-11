"""Concept reward computation utilities."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .schema import ConceptSpec

DEFAULT_WEIGHTS = {
    "match": 0.4,
    "selectivity": 0.3,
    "parsimony": 0.2,
    "transfer": 0.1,
}


def _compute_match(trace_json: Mapping[str, Any], concept: ConceptSpec) -> float:
    expected = set(concept.expected_substructures)
    if not expected:
        return 1.0
    tags = {
        tag
        for feature in trace_json.get("features", [])
        for tag in feature.get("tags", [])
    }
    if not tags:
        return 0.0
    return len(tags & expected) / len(expected)


def _compute_selectivity(trace_json: Mapping[str, Any], target_tags: set[str]) -> float:
    if not target_tags:
        return 1.0
    importance_target = 0.0
    importance_other = 0.0
    for feature in trace_json.get("features", []):
        weight = float(feature.get("importance", 0.0) or 0.0)
        if set(feature.get("tags", [])) & target_tags:
            importance_target += weight
        else:
            importance_other += weight
    total = importance_target + importance_other
    if total == 0:
        return 0.0
    return importance_target / total


def _compute_parsimony(trace_json: Mapping[str, Any], target_tags: set[str]) -> float:
    lengths = trace_json.get("path_lengths", {})
    mean_length = float(lengths.get("mean", 0.0) or 0.0)
    if mean_length <= 0:
        return 0.0
    # Encourage shorter paths when target features dominate.
    target_weight = sum(
        float(edge.get("weight", 0.0) or 0.0)
        for edge in trace_json.get("edges", [])
        if edge.get("src") in target_tags or edge.get("dst") in target_tags
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
) -> float:
    """Return a concept reward combining match, selectivity, parsimony and transfer."""
    if task_metrics is None:
        task_metrics = {}
    if weights is None:
        weights = DEFAULT_WEIGHTS
    target_tags = set(concept_spec.expected_substructures)
    match = _compute_match(trace_json, concept_spec)
    selectivity = _compute_selectivity(trace_json, target_tags)
    parsimony = _compute_parsimony(trace_json, target_tags)
    transfer = _compute_transfer(task_metrics)
    reward = (
        weights.get("match", 0.0) * match
        + weights.get("selectivity", 0.0) * selectivity
        + weights.get("parsimony", 0.0) * parsimony
        + weights.get("transfer", 0.0) * transfer
    )
    return float(reward)


__all__ = ["compute_concept_reward", "DEFAULT_WEIGHTS"]
