"""Artifact helpers for recursive-refinement metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .types import RefinementRun, TrajectoryResult


def summarize_trajectory(trajectory: TrajectoryResult) -> dict[str, Any]:
    """Return compact public metadata suitable for overwatch and logs."""

    final_state = trajectory.states[-1] if trajectory.states else None
    route = []
    if final_state is not None:
        route = [operation.view_name for operation in final_state.operations]
    return {
        "trajectory_id": trajectory.trajectory_id,
        "depth": len(trajectory.states) - 1,
        "active_views": list(final_state.active_views if final_state else []),
        "confidence": trajectory.confidence,
        "uncertainty": final_state.uncertainty if final_state else None,
        "constraint_check_status": _last_status(trajectory, "constraint_check"),
        "verification_status": _last_status(trajectory, "verification"),
        "view_route": route,
        "pruned": trajectory.pruned,
        "converged": trajectory.converged,
    }


def summarize_run(run: RefinementRun) -> dict[str, Any]:
    """Return compact public run-level metadata."""

    return {
        "selected_trajectory_id": run.selected_trajectory_id,
        "total_updates": run.total_updates,
        "max_observed_depth": run.max_observed_depth,
        "max_observed_width": run.max_observed_width,
        "convergence_status": run.convergence_detected,
        "budget_status": "exhausted" if run.budget_exhausted else "within_budget",
        "trajectories": [summarize_trajectory(item) for item in run.trajectories],
    }


def iter_view_route_records(run: RefinementRun) -> Iterable[dict[str, Any]]:
    """Yield one compact route record per trajectory."""

    for trajectory in run.trajectories:
        final_state = trajectory.states[-1] if trajectory.states else None
        operations = final_state.operations if final_state is not None else []
        yield {
            "trajectory_id": trajectory.trajectory_id,
            "view_route": [operation.view_name for operation in operations],
            "operations": [operation.as_dict() for operation in operations],
            "total_updates": trajectory.total_updates,
            "converged": trajectory.converged,
            "pruned": trajectory.pruned,
        }


def _last_status(trajectory: TrajectoryResult, view_name: str) -> bool | None:
    if not trajectory.states:
        return None
    matching = [
        operation.constraint_passed
        for operation in trajectory.states[-1].operations
        if operation.view_name == view_name
    ]
    return matching[-1] if matching else None


__all__ = ["iter_view_route_records", "summarize_run", "summarize_trajectory"]
