import pytest

from rg_tracer.concepts import ConceptSpec
from rg_tracer.runners.self_play import Candidate, _compute_attr_metrics

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
    result = _compute_attr_metrics(candidate, graphs, bonuses, concept=concept)
    assert result["delta_alignment"] > 0
    assert candidate.attr_bonus == pytest.approx(0.05)
    assert candidate.composite == pytest.approx(0.55)


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
        passes_gates=True,
        failed_gates={},
        concept_reward=0.0,
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
    base_reward = candidate.concept_reward
    _compute_attr_metrics(candidate, graphs, bonuses, concept=concept)
    assert candidate.concept_reward >= base_reward
    assert candidate.concept_reward > 0
