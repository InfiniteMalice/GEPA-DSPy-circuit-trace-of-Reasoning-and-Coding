"""Optional auxiliary scores for public reasoning graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .schema import ReasoningGraph
from .validators import SUPPORT_RELATIONS, validate_graph


@dataclass
class GraphScoreReport:
    """Structured score report for graph-native reasoning metadata."""

    validity_score: float
    support_coverage_score: float
    contradiction_penalty: float
    parsimony_score: float
    answer_alignment_score: float
    composite_graph_score: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def score_reasoning_graph(
    graph: ReasoningGraph,
    answer_terms: List[str] | None = None,
) -> GraphScoreReport:
    """Score graph metadata without replacing existing GEPA rubric scoring."""

    validation = validate_graph(graph)
    validity = 1.0 if validation.valid else max(0.0, 1.0 - 0.25 * len(validation.errors))
    final_ids = {node.id for node in graph.nodes if node.kind in {"claim", "conclusion"}}
    supported = {
        edge.target
        for edge in graph.edges
        if edge.target in final_ids and edge.relation in SUPPORT_RELATIONS
    }
    support = len(supported) / len(final_ids) if final_ids else 1.0
    contradictions = sum(1 for edge in graph.edges if edge.relation == "contradicts")
    contradiction_penalty = min(1.0, contradictions / max(1, len(graph.edges)))
    size = len(graph.nodes) + len(graph.edges)
    parsimony = 1.0 if size <= 12 else max(0.0, 1.0 - (size - 12) / 40.0)
    answer_alignment = _answer_alignment(graph, answer_terms or [])
    composite = (
        0.35 * validity
        + 0.25 * support
        + 0.20 * parsimony
        + 0.20 * answer_alignment
        - 0.20 * contradiction_penalty
    )
    return GraphScoreReport(
        validity_score=round(validity, 4),
        support_coverage_score=round(support, 4),
        contradiction_penalty=round(contradiction_penalty, 4),
        parsimony_score=round(parsimony, 4),
        answer_alignment_score=round(answer_alignment, 4),
        composite_graph_score=round(max(0.0, min(1.0, composite)), 4),
        diagnostics={
            "validation_errors": validation.errors,
            "validation_warnings": validation.warnings,
            "cycles": validation.cycles,
        },
    )


def _answer_alignment(graph: ReasoningGraph, answer_terms: List[str]) -> float:
    if not answer_terms:
        return 1.0
    conclusion_text = " ".join(
        node.text.lower() for node in graph.nodes if node.kind in {"claim", "conclusion"}
    )
    hits = sum(1 for term in answer_terms if term.lower() in conclusion_text)
    return hits / len(answer_terms)


__all__ = ["GraphScoreReport", "score_reasoning_graph"]
