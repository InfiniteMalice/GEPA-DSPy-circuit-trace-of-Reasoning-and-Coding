from rg_tracer.hierarchical_state import FoldedState, Subgoal, fold_subgoals


def test_completed_subgoal_folds_into_summary():
    state = fold_subgoals("t1", [Subgoal("s1", "t1", "done", "completed")])
    assert "done" in state.completed_summary


def test_unresolved_constraint_is_not_dropped():
    state = fold_subgoals("t1", [Subgoal("s1", "t1", "active", "active", [], ["keep"])])
    assert state.unresolved_constraints == ["keep"]


def test_failed_attempt_is_preserved():
    state = fold_subgoals("t1", [Subgoal("s1", "t1", "bad path", "blocked")])
    assert state.failed_attempts == ["bad path"]


def test_folded_state_reloads():
    state = FoldedState("t1", "summary", unresolved_constraints=["keep"])
    reloaded = FoldedState.from_json(state.to_json())
    assert reloaded.to_dict() == state.to_dict()
