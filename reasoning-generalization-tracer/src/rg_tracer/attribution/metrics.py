"""Metric helpers for attribution graphs."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Mapping, Sequence

from .schema import AttributionGraph, GraphEdge, normalise_graph
from .utils import safe_float


GraphLike = Mapping[str, object] | AttributionGraph


def _ensure_graphs(graphs: Iterable[GraphLike]) -> List[AttributionGraph]:
    result: List[AttributionGraph] = []
    for graph in graphs:
        if isinstance(graph, AttributionGraph):
            result.append(graph)
        else:
            result.append(normalise_graph(graph))
    return result


def _edge_weights(graph: AttributionGraph) -> List[float]:
    weights: List[float] = []
    for edge in graph.edges:
        weight = safe_float(edge.attr)
        if abs(weight) > 0.0:
            weights.append(weight)
    return weights


def path_sparsity(
    graphs: Sequence[GraphLike] | GraphLike,
    *,
    eps: float = 1e-9,
) -> float:
    """Return mean ``HHI = Σ p_i^2`` for normalised edge attributions.

    ``eps`` is applied only when the attribution mass is extremely small to keep
    the denominator numerically stable without biasing typical totals.
    """

    graphs_seq = _ensure_graphs(_coerce_sequence(graphs))
    if not graphs_seq:
        return 0.0
    scores: List[float] = []
    for graph in graphs_seq:
        weights = _edge_weights(graph)
        total = sum(abs(weight) for weight in weights)
        if total <= 0:
            scores.append(0.0)
            continue
        denom = total if total > eps else total + eps
        normed = [abs(weight) / denom for weight in weights]
        scores.append(sum(value * value for value in normed))
    return float(sum(scores) / len(scores))


def average_path_length(graphs: Sequence[GraphLike] | GraphLike) -> float:
    """Return expected hop distance ``E[len]`` weighted by edge attribution."""

    graphs_seq = _ensure_graphs(_coerce_sequence(graphs))
    if not graphs_seq:
        return 0.0
    lengths: List[float] = []
    for graph in graphs_seq:
        layers: dict[str, int] = {}
        for node in graph.nodes:
            raw_layer = getattr(node, "layer", 0)
            try:
                layer_value = int(raw_layer)
            except (TypeError, ValueError):
                layer_value = 0
            layers[node.id] = layer_value
        total_weight = 0.0
        weighted_length = 0.0
        for edge in graph.edges:
            weight = abs(safe_float(edge.attr))
            if weight <= 0:
                continue
            src_layer = layers.get(edge.src)
            if src_layer is None:
                src_layer = 0
            dst_layer = layers.get(edge.dst)
            if dst_layer is None:
                dst_layer = src_layer + 1
            hop = max(1, dst_layer - src_layer)
            weighted_length += hop * weight
            total_weight += weight
        if total_weight <= 0:
            lengths.append(0.0)
        else:
            lengths.append(weighted_length / total_weight)
    return float(sum(lengths) / len(lengths))


def average_branching_factor(
    graphs: Sequence[GraphLike] | GraphLike,
    *,
    top_k: int | None = None,
) -> float:
    """Return mean branching factor ``Σ_k out_degree_k * w_k / Σ_k w_k``."""

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    graphs_seq = _ensure_graphs(_coerce_sequence(graphs))
    if not graphs_seq:
        return 0.0
    factors: List[float] = []
    for graph in graphs_seq:
        outgoing: dict[str, List[GraphEdge]] = {}
        for edge in graph.edges:
            outgoing.setdefault(edge.src, []).append(edge)
        if not outgoing:
            factors.append(0.0)
            continue
        weighted_sum = 0.0
        total_weight = 0.0
        for edges in outgoing.values():
            sorted_edges = sorted(edges, key=lambda e: abs(safe_float(e.attr)), reverse=True)
            if top_k is not None:
                sorted_edges = sorted_edges[:top_k]
            weight = sum(abs(safe_float(edge.attr)) for edge in sorted_edges)
            if weight <= 0:
                continue
            weighted_sum += len(sorted_edges) * weight
            total_weight += weight
        if total_weight <= 0:
            factors.append(0.0)
        else:
            factors.append(weighted_sum / total_weight)
    return float(sum(factors) / len(factors))


def repeatability(
    graphs: Sequence[GraphLike] | GraphLike,
    *,
    top_k: int = 10,
) -> float:
    """Return mean weighted Jaccard overlap over edge attributions."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    graphs_seq = _ensure_graphs(_coerce_sequence(graphs))
    if len(graphs_seq) <= 1:
        return 1.0
    overlaps: List[float] = []
    weight_maps = [_edge_weight_map(graph, top_k=top_k) for graph in graphs_seq]
    for left, right in combinations(weight_maps, 2):
        keys = set(left) | set(right)
        if not keys:
            overlaps.append(1.0)
            continue
        numerator = sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
        denominator = sum(max(left.get(key, 0.0), right.get(key, 0.0)) for key in keys)
        if denominator <= 0:
            overlaps.append(1.0)
        else:
            overlaps.append(numerator / denominator)
    return float(sum(overlaps) / len(overlaps))


def concept_alignment(
    graphs: Sequence[GraphLike] | GraphLike,
    concept_features: Sequence[Mapping[str, object]] | None,
    *,
    top_k: int = 10,
) -> float:
    """Return overlap@k combining node matches and edge attribution mass."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not concept_features:
        return 0.0
    graphs_seq = _ensure_graphs(_coerce_sequence(graphs))
    if not graphs_seq:
        return 0.0
    concept_ids = {
        str(feature["id"])
        for feature in concept_features
        if "id" in feature and feature["id"] is not None
    }
    if not concept_ids:
        return 0.0
    scores: List[float] = []
    for graph in graphs_seq:
        node_ids = _top_node_ids(graph, top_k=top_k)
        node_denom = max(1, min(len(node_ids), len(concept_ids)))
        node_score = len(node_ids & concept_ids) / node_denom
        weight_map = _edge_weight_map(graph, top_k=top_k)
        weight_total = sum(weight_map.values())
        if weight_total <= 0:
            edge_score = 0.0
        else:
            edge_score = (
                sum(
                    weight
                    for key, weight in weight_map.items()
                    if any(node in concept_ids for node in key.split("->"))
                )
                / weight_total
            )
        scores.append((node_score + edge_score) / 2.0)
    return float(sum(scores) / len(scores))


def delta_sparsity(
    graphs: Sequence[GraphLike] | GraphLike,
    *,
    overfit_phase: str = "overfit",
    post_phase: str = "post_grok",
) -> float:
    """Return ``Δ = sparsity_overfit - sparsity_post`` (positive = flatter paths)."""

    graphs_seq = _ensure_graphs(_coerce_sequence(graphs))
    overfit_graphs = _filter_by_phase(graphs_seq, overfit_phase)
    post_graphs = _filter_by_phase(graphs_seq, post_phase)
    if not overfit_graphs or not post_graphs:
        return 0.0
    overfit = path_sparsity(overfit_graphs)
    post = path_sparsity(post_graphs)
    return float(overfit - post)


def delta_alignment(
    graphs: Sequence[GraphLike] | GraphLike,
    concept_features: Sequence[Mapping[str, object]] | None,
    *,
    overfit_phase: str = "overfit",
    post_phase: str = "post_grok",
    top_k: int = 10,
) -> float:
    """Return ``Δ = alignment_post - alignment_overfit``."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    graphs_seq = _ensure_graphs(_coerce_sequence(graphs))
    overfit_graphs = _filter_by_phase(graphs_seq, overfit_phase)
    post_graphs = _filter_by_phase(graphs_seq, post_phase)
    if not overfit_graphs or not post_graphs:
        return 0.0
    overfit = concept_alignment(overfit_graphs, concept_features, top_k=top_k)
    post = concept_alignment(post_graphs, concept_features, top_k=top_k)
    return float(post - overfit)


def delta_repeatability(
    graphs: Sequence[GraphLike] | GraphLike,
    *,
    overfit_phase: str = "overfit",
    post_phase: str = "post_grok",
    top_k: int = 10,
) -> float:
    """Return ``Δ = repeat_post - repeat_overfit``."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    graphs_seq = _ensure_graphs(_coerce_sequence(graphs))
    overfit_graphs = _filter_by_phase(graphs_seq, overfit_phase)
    post_graphs = _filter_by_phase(graphs_seq, post_phase)
    if not overfit_graphs or not post_graphs:
        return 0.0
    overfit = repeatability(overfit_graphs, top_k=top_k)
    post = repeatability(post_graphs, top_k=top_k)
    return float(post - overfit)


def _coerce_sequence(graphs: Sequence[GraphLike] | GraphLike | None) -> List[GraphLike]:
    if graphs is None:
        return []
    if isinstance(graphs, (AttributionGraph, Mapping)):
        return [graphs]
    return [graph for graph in graphs if graph is not None]


def _filter_by_phase(
    graphs: Sequence[GraphLike] | GraphLike,
    phase: str,
) -> List[GraphLike]:
    filtered: List[GraphLike] = []
    for graph in _ensure_graphs(_coerce_sequence(graphs)):
        if graph.meta.phase == phase:
            filtered.append(graph)
    return filtered


def _top_node_ids(graph: AttributionGraph, *, top_k: int) -> set[str]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ranked = sorted(
        graph.nodes,
        key=lambda node: abs(safe_float(getattr(node, "activation", 0.0))),
        reverse=True,
    )
    ranked = ranked[:top_k]
    return {str(node.id) for node in ranked}


def _edge_weight_map(graph: AttributionGraph, *, top_k: int) -> dict[str, float]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ranked = sorted(graph.edges, key=lambda e: abs(safe_float(e.attr)), reverse=True)
    ranked = ranked[:top_k]
    total = sum(abs(safe_float(edge.attr)) for edge in ranked)
    if total <= 0:
        return {}
    return {f"{edge.src}->{edge.dst}": abs(safe_float(edge.attr)) / total for edge in ranked}


__all__ = [
    "average_branching_factor",
    "average_path_length",
    "concept_alignment",
    "delta_alignment",
    "delta_repeatability",
    "delta_sparsity",
    "path_sparsity",
    "repeatability",
]
