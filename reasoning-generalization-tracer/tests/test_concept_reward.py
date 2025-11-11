import pytest

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


def test_alignment_boost_scales_reward():
    spec = ConceptSpec(name="demo", definition="", expected_substructures=["edge"])
    trace = {
        "features": [{"id": "edge", "layer": 0, "importance": 0.5, "tags": ["edge"]}],
        "edges": [],
        "path_lengths": {"mean": 1.0},
    }
    base = compute_concept_reward(trace, spec, task_metrics={})
    boosted = compute_concept_reward(
        trace,
        spec,
        task_metrics={},
        alignment=0.8,
    )
    assert boosted > base


def _basic_trace_spec() -> tuple[ConceptSpec, dict[str, object]]:
    spec = ConceptSpec(name="edge", definition="", expected_substructures=["edge"])
    trace = {
        "features": [{"id": "edge", "layer": 0, "importance": 0.5, "tags": ["edge"]}],
        "edges": [],
        "path_lengths": {"mean": 1.0},
    }
    return spec, trace


def test_alignment_zero_leaves_reward_unchanged():
    spec, trace = _basic_trace_spec()
    base = compute_concept_reward(trace, spec, task_metrics={})
    zero = compute_concept_reward(trace, spec, task_metrics={}, alignment=0.0)
    assert zero == pytest.approx(base)


def test_alignment_negative_is_neutral():
    spec, trace = _basic_trace_spec()
    base = compute_concept_reward(trace, spec, task_metrics={})
    negative = compute_concept_reward(trace, spec, task_metrics={}, alignment=-0.3)
    assert negative == pytest.approx(base)


def test_alignment_positive_scales_reward():
    spec, trace = _basic_trace_spec()
    base = compute_concept_reward(trace, spec, task_metrics={})
    scaled = compute_concept_reward(
        trace,
        spec,
        task_metrics={},
        alignment=1.0,
        alignment_scale=0.5,
    )
    assert scaled == pytest.approx(base * 1.5)


def test_alignment_clamps_to_one():
    spec, trace = _basic_trace_spec()
    base = compute_concept_reward(trace, spec, task_metrics={})
    clamped = compute_concept_reward(
        trace,
        spec,
        task_metrics={},
        alignment=5.0,
        alignment_scale=0.5,
    )
    assert clamped == pytest.approx(base * 1.5)


def test_alignment_none_returns_base_reward():
    spec, trace = _basic_trace_spec()
    base = compute_concept_reward(trace, spec, task_metrics={})
    none_value = compute_concept_reward(trace, spec, task_metrics={}, alignment=None)
    assert none_value == pytest.approx(base)


def test_alignment_zero_scale_preserves_reward():
    spec, trace = _basic_trace_spec()
    base = compute_concept_reward(trace, spec, task_metrics={})
    zero_scale = compute_concept_reward(
        trace,
        spec,
        task_metrics={},
        alignment=0.8,
        alignment_scale=0.0,
    )
    assert zero_scale == pytest.approx(base)


def test_alignment_penalty_applies_before_multiplier():
    spec, trace = _basic_trace_spec()
    penalty_reward = compute_concept_reward(
        trace,
        spec,
        task_metrics={"contradictory_feature_ids": ["edge"]},
        alignment=0.9,
        alignment_scale=0.5,
    )
    boosted_clean = compute_concept_reward(
        trace,
        spec,
        task_metrics={},
        alignment=0.9,
        alignment_scale=0.5,
    )
    assert penalty_reward < boosted_clean
