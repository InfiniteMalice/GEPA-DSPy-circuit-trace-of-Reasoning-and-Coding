"""Information folding helpers."""

from __future__ import annotations

from typing import Iterable

from .state_summary import FoldedState
from .subgoals import Subgoal


def fold_subgoals(task_id: str, subgoals: Iterable[Subgoal]) -> FoldedState:
    completed: list[str] = []
    active: list[dict[str, object]] = []
    constraints: list[str] = []
    failed: list[str] = []
    evidence: list[str] = []
    for subgoal in subgoals:
        constraints.extend(subgoal.open_constraints)
        evidence.extend(subgoal.evidence_refs)
        if subgoal.status == "completed":
            completed.append(subgoal.description)
        elif subgoal.status in {"blocked", "abandoned"}:
            failed.append(subgoal.description)
        else:
            active.append(subgoal.to_dict())
    return FoldedState(
        task_id=task_id,
        completed_summary="; ".join(completed),
        active_subgoals=active,
        unresolved_constraints=constraints,
        failed_attempts=failed,
        evidence_refs=evidence,
    )


__all__ = ["fold_subgoals"]
