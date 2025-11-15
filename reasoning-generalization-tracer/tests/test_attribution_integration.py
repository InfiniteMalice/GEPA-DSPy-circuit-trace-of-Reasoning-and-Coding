import pytest

from rg_tracer.concepts import ConceptSpec, compute_concept_reward
from rg_tracer.runners.self_play import Candidate, _compute_and_apply_attr_metrics

BASE_GRAPH = {
    "model_ref": "m",
    "task_id": "t",
    "nodes": [
        {"id": "n0", "layer": 0, "type": "token", "activation": 1.0},
        {"id": "n1", "layer": 1, "type": "mlp", "activation": 0.5},
        {"id": "n2", "layer": 2, "type": "logit", "activation": 0.6},
    ],
}


def _phase_graph(phase, weight):
    edges = [
        {"src": "n0", "dst": "n1", "attr": 0.2},
        {"src": "n1", "dst": "n2", "attr": weight},
    ]
    meta = {"phase": phase}
    return {**BASE_GRAPH, "edges": edges, "meta": meta}


def test_attr_bonus_applies_when_alignment_grows():
    concept = ConceptSpec(name="demo", definition="", expected_substructures=["n2"])
    candidate = Candidate(
        text="",
        confidence=0.9,
        metrics={},
        axis_scores={},
        composite=0.5,
        base_composite=0.5,
        passes_gates=True,
        failed_gates={},
        concept_reward=0.1,
        abstained=False,
        trace={
            "features": [{"id": "n2", "layer": 2, "importance": 0.5, "tags": ["n2"]}],
            "edges": [],
            "path_lengths": {"mean": 1.0},
        },
        problem_id="p",
        semantic_report={},
        semantics_map={"entailed_feature_ids": ["n2"], "contradictory_feature_ids": []},
    )
    graphs = [_phase_graph("overfit", 0.1), _phase_graph("post_grok", 0.9)]
    bonuses = {
        "alignment_gain": 0.05,
        "repeatability_gain": 0.0,
        "sparsity_drop": 0.0,
    }
    result = _compute_and_apply_attr_metrics(candidate, graphs, bonuses, concept=concept)
    assert result["delta_alignment"] > 0
    assert candidate.attr_bonus == pytest.approx(0.05)
    assert candidate.composite == pytest.approx(0.55)
    assert {
        "delta_alignment",
        "delta_repeatability",
        "delta_sparsity",
    }.issubset(candidate.attr_metrics.keys())
    assert all(isinstance(value, float) for value in candidate.attr_metrics.values())


def test_concept_reward_scales_with_alignment():
    concept = ConceptSpec(
        name="demo",
        definition="",
        expected_substructures=["n2"],
    )
    candidate = Candidate(
        text="",
        confidence=0.9,
        metrics={},
        axis_scores={},
        composite=0.5,
        base_composite=0.5,
        passes_gates=True,
        failed_gates={},
        concept_reward=0.2,
        abstained=False,
        trace={
            "features": [{"id": "n2", "layer": 2, "importance": 1.0, "tags": ["n2"]}],
            "edges": [
                {"src": "n1", "dst": "n2", "weight": 1.0},
            ],
            "path_lengths": {"mean": 1.0},
        },
        problem_id="p",
        semantic_report={},
        semantics_map={
            "entailed_feature_ids": ["n2"],
            "contradictory_feature_ids": [],
        },
    )
    graphs = [_phase_graph("overfit", 0.1), _phase_graph("post_grok", 0.9)]
    bonuses = {
        "alignment_gain": 0.0,
        "repeatability_gain": 0.0,
        "sparsity_drop": 0.0,
    }
    task_metrics = {
        **candidate.semantics_map,
        "concept_reuse": 1.0,
        "supporting_tasks": 1.0,
    }
    base_reward = compute_concept_reward(
        candidate.trace,
        concept,
        task_metrics=task_metrics,
        alignment=None,
    )
    metrics = _compute_and_apply_attr_metrics(candidate, graphs, bonuses, concept=concept)
    alignment_value = metrics.get("alignment") if metrics else None
    aligned_reward = compute_concept_reward(
        candidate.trace,
        concept,
        task_metrics=task_metrics,
        alignment=alignment_value,
    )
    assert aligned_reward > base_reward
