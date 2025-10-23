from rg_tracer.concepts import ConceptSpec, compute_concept_reward


def test_concept_reward_increases_with_semantic_matches():
    spec = ConceptSpec(
        name="parity", definition="Detects parity", expected_substructures=["parity"]
    )
    trace = {
        "features": [
            {"id": "F0", "layer": 0, "importance": 0.2, "tags": ["baseline"]},
            {"id": "F1", "layer": 1, "importance": 0.6, "tags": ["parity"]},
        ],
        "edges": [{"src": "F0", "dst": "F1", "weight": 0.5}],
        "path_lengths": {"mean": 2.0},
    }
    low_reward = compute_concept_reward(
        trace,
        spec,
        task_metrics={
            "concept_reuse": 0.2,
            "supporting_tasks": 1,
            "entailed_feature_ids": ["F0"],
        },
    )
    high_reward = compute_concept_reward(
        trace,
        spec,
        task_metrics={
            "concept_reuse": 0.9,
            "supporting_tasks": 1,
            "entailed_feature_ids": ["F1"],
        },
    )
    penalised = compute_concept_reward(
        trace,
        spec,
        task_metrics={
            "concept_reuse": 0.9,
            "supporting_tasks": 1,
            "entailed_feature_ids": ["F1"],
            "contradictory_feature_ids": ["F1"],
        },
    )
    assert high_reward > low_reward
    assert penalised < high_reward
