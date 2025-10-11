from rg_tracer.concepts import ConceptSpec, compute_concept_reward


def test_concept_reward_increases_with_matches():
    spec = ConceptSpec(name="parity", definition="Detects parity", expected_substructures=["parity"])
    trace_low = {
        "features": [
            {"id": "F0", "layer": 0, "importance": 0.2, "tags": ["baseline"]}
        ],
        "edges": [],
        "path_lengths": {"mean": 4.0},
    }
    trace_high = {
        "features": [
            {"id": "F1", "layer": 1, "importance": 0.6, "tags": ["parity"]}
        ],
        "edges": [{"src": "parity", "dst": "other", "weight": 0.5}],
        "path_lengths": {"mean": 2.0},
    }
    reward_low = compute_concept_reward(trace_low, spec, task_metrics={"concept_reuse": 0.2, "supporting_tasks": 1})
    reward_high = compute_concept_reward(trace_high, spec, task_metrics={"concept_reuse": 0.9, "supporting_tasks": 1})
    assert reward_high > reward_low
