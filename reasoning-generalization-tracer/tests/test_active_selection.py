import pytest

from rg_tracer.active_selection import disagreement_score, select_diagnostic_tasks


def test_disagreement_score_increases_when_candidates_differ():
    assert disagreement_score(["a", "b", "c"]) > disagreement_score(["a", "a", "a"])


def test_top_k_selection_is_deterministic():
    tasks = [
        {"task_id": "b", "predictions": ["x", "x"]},
        {"task_id": "a", "predictions": ["x", "y"]},
    ]
    selected = select_diagnostic_tasks(tasks, top_k=1)
    assert selected[0]["task_id"] == "a"


def test_empty_candidates_handled_safely():
    selected = select_diagnostic_tasks([{"task_id": "a", "predictions": []}], top_k=1)
    assert selected[0]["disagreement_score"] == 0.0
    assert selected[0]["metadata"]["reason"] == "candidate predictions agree"


def test_negative_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k"):
        select_diagnostic_tasks([], top_k=-1)
