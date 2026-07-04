"""Validation helpers for public reasoning graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from .schema import ReasoningGraph

FINAL_KINDS = {"claim", "conclusion"}
SUPPORT_RELATIONS = {"supports", "tests", "implements", "depends_on"}
CONTRADICTION_RELATIONS = {"contradicts"}


@dataclass
class GraphValidationResult:
    """Structured graph validation diagnostics."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    cycles: List[List[str]] = field(default_factory=list)


def detect_cycles(graph: ReasoningGraph) -> List[List[str]]:
    adjacency: Dict[str, List[str]] = {}
    for edge in graph.edges:
        adjacency.setdefault(edge.source, []).append(edge.target)
    cycles: List[List[str]] = []
    visited: Set[str] = set()

    for node in graph.nodes:
        if node.id in visited:
            continue
        path: List[str] = []
        path_index: Dict[str, int] = {}
        stack = [(node.id, iter(adjacency.get(node.id, [])))]
        while stack:
            current, children = stack[-1]
            if current not in path_index:
                path_index[current] = len(path)
                path.append(current)
            try:
                target = next(children)
            except StopIteration:
                stack.pop()
                path_index.pop(current, None)
                if path and path[-1] == current:
                    path.pop()
                visited.add(current)
                continue
            if target in path_index:
                index = path_index[target]
                cycles.append(path[index:] + [target])
            elif target not in visited:
                stack.append((target, iter(adjacency.get(target, []))))
    return cycles


def validate_graph(graph: ReasoningGraph, final_answer_task: bool = True) -> GraphValidationResult:
    """Validate graph structure without rejecting intentional contradictions."""

    errors: List[str] = []
    warnings: List[str] = []
    node_ids = [node.id for node in graph.nodes]
    unique_ids = set(node_ids)
    if len(node_ids) != len(unique_ids):
        errors.append("node IDs must be unique")
    for edge in graph.edges:
        if edge.source not in unique_ids or edge.target not in unique_ids:
            errors.append(f"edge endpoint missing for {edge.source!r}->{edge.target!r}")
        if not edge.relation:
            errors.append("relation labels must be nonempty")
        if edge.relation in CONTRADICTION_RELATIONS:
            warnings.append(f"contradictory edge surfaced: {edge.source}->{edge.target}")
    if final_answer_task:
        final_nodes = [node for node in graph.nodes if node.kind in FINAL_KINDS]
        if not final_nodes:
            errors.append("final-answer tasks need at least one claim or conclusion node")
        unsupported = [
            node.id
            for node in final_nodes
            if not any(
                edge.target == node.id
                and (
                    edge.relation not in CONTRADICTION_RELATIONS
                    and (
                        edge.relation in SUPPORT_RELATIONS
                        or graph_node_kind(graph, edge.source) in {"evidence", "test"}
                    )
                )
                for edge in graph.edges
            )
        ]
        if unsupported:
            errors.append(f"final nodes without support, evidence, or test paths: {unsupported}")
    cycles = detect_cycles(graph)
    if cycles:
        warnings.append("cycles detected")
    return GraphValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        cycles=cycles,
    )


def graph_node_kind(graph: ReasoningGraph, node_id: str) -> str:
    for node in graph.nodes:
        if node.id == node_id:
            return node.kind
    return ""


__all__ = ["GraphValidationResult", "detect_cycles", "validate_graph"]
