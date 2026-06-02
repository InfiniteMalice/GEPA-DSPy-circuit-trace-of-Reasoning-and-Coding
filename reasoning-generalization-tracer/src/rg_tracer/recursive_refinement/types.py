"""Public JSON-safe trajectory metadata for recursive refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class ViewOperation:
    """A public operator-level record, excluding unrestricted hidden reasoning text."""

    view_name: str
    operation_name: str
    step_index: int
    rationale: str
    revisited: bool = False
    changed_prediction: bool = False
    reduced_uncertainty: bool = False
    increased_confidence: bool = False
    constraint_passed: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class RefinementState:
    """A public state snapshot for one trajectory."""

    trajectory_id: str
    parent_id: str | None
    depth: int
    state_value: float
    uncertainty: float
    confidence: float
    prediction: object | None
    active_views: list[str]
    operations: list[ViewOperation]
    halted: bool = False
    pruned: bool = False
    halt_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class TrajectoryResult:
    """Final public summary for a candidate trajectory."""

    trajectory_id: str
    states: list[RefinementState]
    prediction: object | None
    confidence: float
    process_score: float
    total_updates: int
    converged: bool
    pruned: bool
    process_score_components: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class RefinementRun:
    """Public summary for a full recursive-refinement run."""

    trajectories: list[TrajectoryResult]
    selected_trajectory_id: str
    total_updates: int
    max_observed_depth: int
    max_observed_width: int
    convergence_detected: bool
    budget_exhausted: bool

    def as_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))
