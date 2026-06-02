"""Adaptive recursive-refinement controller for the experimental scaffold."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Mapping

from ..trm_baseline import TinyRecursionModel
from .config import RecursiveRefinementConfig
from .scoring import score_trajectory
from .types import RefinementRun, RefinementState, TrajectoryResult
from .views import ReasoningView, build_default_view_registry


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


class RecursiveRefinementController:
    """Seeded CPU-friendly recursive refinement with progressive widening."""

    def __init__(
        self,
        config: RecursiveRefinementConfig | None = None,
        *,
        model: TinyRecursionModel | None = None,
        views: Mapping[str, ReasoningView] | None = None,
    ) -> None:
        self.config = config or RecursiveRefinementConfig()
        self.model = model or TinyRecursionModel()
        self.views = dict(views or build_default_view_registry())
        self.rng = random.Random(self.config.budget.seed)

    def run(
        self,
        problem: Mapping[str, object],
        *,
        target_width: int | None = None,
    ) -> RefinementRun:
        """Run the bounded refinement loop and return public trajectory metadata."""

        budget = self.config.budget
        max_width = min(budget.max_width, target_width or budget.max_width)
        max_width = max(1, max_width)
        root = self._initial_state(problem)
        histories: dict[str, list[RefinementState]] = {root.trajectory_id: [root]}
        active: list[RefinementState] = [root]
        total_updates = 1
        max_observed_width = len(active)

        initial_target_width = min(max_width, budget.initial_width)
        if initial_target_width > 1:
            active, total_updates = self._widen(
                active,
                histories,
                total_updates=total_updates,
                max_width=initial_target_width,
            )
            max_observed_width = max(max_observed_width, len(active))
        elif self._should_widen(active) and max_width > 1:
            active, total_updates = self._widen(
                active,
                histories,
                total_updates=total_updates,
                max_width=max_width,
            )
            max_observed_width = max(max_observed_width, len(active))

        while active and total_updates < budget.max_total_updates:
            next_active: list[RefinementState] = []
            for state in active:
                if total_updates >= budget.max_total_updates:
                    next_active.append(self._halt(state, "update_budget_exhausted"))
                    break
                if state.halted or state.pruned:
                    next_active.append(state)
                    continue
                if self._should_halt(state, active):
                    halted = self._halt(state, "adaptive_confidence_or_convergence")
                    histories[state.trajectory_id][-1] = halted
                    next_active.append(halted)
                    continue
                view = self._select_view(state, problem)
                if view is None:
                    halted = self._halt(state, "no_relevant_view")
                    histories[state.trajectory_id][-1] = halted
                    next_active.append(halted)
                    continue
                refined = view.apply(state, problem)
                total_updates += 1
                if refined.state_value < budget.prune_below:
                    refined = replace(
                        refined,
                        pruned=True,
                        halted=True,
                        halt_reason="pruned_below_threshold",
                    )
                elif refined.depth >= budget.max_depth:
                    refined = self._halt(refined, "max_depth_reached")
                histories[refined.trajectory_id].append(refined)
                next_active.append(refined)
            active = [state for state in next_active if not state.pruned]
            active = active[:max_width]
            if not active:
                break
            if (
                self.config.progressive_widening
                and len(active) < max_width
                and total_updates < budget.max_total_updates
                and self._should_widen(active)
            ):
                active, total_updates = self._widen(
                    active,
                    histories,
                    total_updates=total_updates,
                    max_width=max_width,
                )
            max_observed_width = max(max_observed_width, len(active))
            if all(state.halted or state.pruned for state in active):
                break

        budget_exhausted = total_updates >= budget.max_total_updates
        leaves = [history[-1] for history in histories.values() if history]
        convergence_detected = self._converged(leaves)
        trajectories = self._build_results(
            histories,
            total_updates=total_updates,
            budget_exhausted=budget_exhausted,
            convergence_detected=convergence_detected,
        )
        selected = max(
            trajectories,
            key=lambda trajectory: (trajectory.process_score, trajectory.confidence),
        )
        max_observed_depth = max(
            (state.depth for trajectory in trajectories for state in trajectory.states),
            default=0,
        )
        return RefinementRun(
            trajectories=trajectories,
            selected_trajectory_id=selected.trajectory_id,
            total_updates=total_updates,
            max_observed_depth=max_observed_depth,
            max_observed_width=max_observed_width,
            convergence_detected=convergence_detected,
            budget_exhausted=budget_exhausted,
        )

    def _initial_state(self, problem: Mapping[str, object]) -> RefinementState:
        prediction = self._initial_prediction(problem)
        confidence = self._initial_confidence(problem, prediction)
        uncertainty = _clamp_probability(1.0 - confidence + 0.08)
        return RefinementState(
            trajectory_id="traj_0",
            parent_id=None,
            depth=0,
            state_value=confidence - uncertainty,
            uncertainty=uncertainty,
            confidence=confidence,
            prediction=prediction,
            active_views=[],
            operations=[],
        )

    def _initial_prediction(self, problem: Mapping[str, object]) -> object | None:
        numbers = problem.get("numbers", [])
        if isinstance(numbers, (list, tuple)) and numbers:
            ints = [
                item for item in numbers if isinstance(item, int) and not isinstance(item, bool)
            ]
            if ints:
                return sum(ints)
        sequence = problem.get("sequence", [])
        if problem.get("task") == "parity" and isinstance(sequence, (list, tuple)):
            ints = [
                item for item in sequence if isinstance(item, int) and not isinstance(item, bool)
            ]
            if ints:
                return sum(ints) % 2
        return 0

    def _initial_confidence(
        self,
        problem: Mapping[str, object],
        prediction: object | None,
    ) -> float:
        sequence = problem.get("sequence") or problem.get("numbers") or []
        if isinstance(sequence, (list, tuple)):
            values = [int(item) for item in sequence if isinstance(item, int)]
            model_confidence = self.model.predict(values[:8]) if values else 0.5
        else:
            model_confidence = 0.5
        label_bonus = 0.10 if problem.get("answer") == prediction else 0.0
        return _clamp_probability(0.45 + 0.20 * model_confidence + label_bonus)

    def _select_view(
        self,
        state: RefinementState,
        problem: Mapping[str, object],
    ) -> ReasoningView | None:
        if not self.config.routing.enabled:
            return self.views.get("verification") or self.views.get("compression")
        eligible = [
            view
            for view in self.views.values()
            if view.applicable(problem)
            and (self.config.routing.revisit_views or view.name not in state.active_views)
        ]
        if not eligible:
            return None
        names = self._prioritized_view_names(state, problem)
        ordered = [
            self.views[name]
            for name in names
            if name in self.views and self.views[name] in eligible
        ]
        ordered.extend(view for view in eligible if view not in ordered)
        active_window = ordered[: self.config.routing.max_active_views]
        if self.config.routing.route_strategy == "fixed":
            return active_window[0]
        if self.config.routing.route_strategy == "round_robin":
            return active_window[state.depth % len(active_window)]
        return active_window[self.rng.randrange(len(active_window))]

    def _prioritized_view_names(
        self,
        state: RefinementState,
        problem: Mapping[str, object],
    ) -> list[str]:
        if problem.get("task") == "parity":
            names = ["parity", "constraint_check", "verification", "counterexample_check"]
        elif state.uncertainty >= self.config.budget.uncertainty_threshold:
            names = ["arithmetic", "constraint_check", "verification", "counterexample_check"]
        else:
            names = ["verification", "compression", "constraint_check", "parity"]
        names.append("compression")
        return names

    def _should_halt(
        self,
        state: RefinementState,
        peers: list[RefinementState],
    ) -> bool:
        if not self.config.adaptive_halting:
            return False
        budget = self.config.budget
        if state.depth < budget.min_depth:
            return False
        confident = (
            state.confidence >= budget.convergence_threshold
            and state.uncertainty <= budget.uncertainty_threshold
        )
        if confident:
            return True
        return self._converged(peers) and state.uncertainty <= budget.uncertainty_threshold

    def _should_widen(self, active: list[RefinementState]) -> bool:
        if not self.config.progressive_widening:
            return False
        if not active:
            return False
        uncertainty = sum(state.uncertainty for state in active) / len(active)
        disagreement = self._disagreement(active)
        budget = self.config.budget
        return (
            uncertainty >= budget.uncertainty_threshold
            or disagreement >= budget.disagreement_threshold
        )

    def _widen(
        self,
        active: list[RefinementState],
        histories: dict[str, list[RefinementState]],
        *,
        total_updates: int,
        max_width: int,
    ) -> tuple[list[RefinementState], int]:
        widened = list(active)
        budget = self.config.budget
        parent_index = 0
        while (
            len(widened) < max_width
            and total_updates < budget.max_total_updates
            and parent_index < budget.branch_factor + len(active)
        ):
            parent = active[parent_index % len(active)]
            child_id = f"traj_{len(histories)}"
            child = replace(
                parent,
                trajectory_id=child_id,
                parent_id=parent.trajectory_id,
                confidence=_clamp_probability(parent.confidence - 0.10),
                uncertainty=_clamp_probability(parent.uncertainty + 0.12),
                prediction=self._perturb_prediction(parent.prediction),
                state_value=parent.state_value - 0.22,
                halted=False,
                pruned=False,
                halt_reason=None,
            )
            histories[child_id] = list(histories[parent.trajectory_id])
            histories[child_id].append(child)
            widened.append(child)
            total_updates += 1
            parent_index += 1
        return widened, total_updates

    def _perturb_prediction(self, prediction: object | None) -> object | None:
        if isinstance(prediction, int) and not isinstance(prediction, bool):
            shift = self.rng.choice([-2, -1, 1, 2])
            return prediction + shift
        return prediction

    def _halt(self, state: RefinementState, reason: str) -> RefinementState:
        return replace(state, halted=True, halt_reason=reason)

    def _disagreement(self, states: list[RefinementState]) -> float:
        predictions = {state.prediction for state in states}
        if len(predictions) <= 1:
            return 0.0
        return 1.0 - (1.0 / len(predictions))

    def _converged(self, states: list[RefinementState]) -> bool:
        if len(states) < 2:
            return False
        active = [state for state in states if not state.pruned]
        if len(active) < 2:
            return False
        predictions = {state.prediction for state in active}
        average_confidence = sum(state.confidence for state in active) / len(active)
        return (
            len(predictions) == 1 and average_confidence >= self.config.budget.convergence_threshold
        )

    def _build_results(
        self,
        histories: Mapping[str, list[RefinementState]],
        *,
        total_updates: int,
        budget_exhausted: bool,
        convergence_detected: bool,
    ) -> list[TrajectoryResult]:
        results = []
        for trajectory_id, states in histories.items():
            if not states:
                continue
            final = states[-1]
            updates = max(0, len(states) - 1)
            process_score = score_trajectory(
                states,
                total_updates=updates,
                max_total_updates=self.config.budget.max_total_updates,
                converged=convergence_detected,
                budget_exhausted=budget_exhausted,
            )
            results.append(
                TrajectoryResult(
                    trajectory_id=trajectory_id,
                    states=list(states) if self.config.log_trajectory_states else [final],
                    prediction=final.prediction,
                    confidence=final.confidence,
                    process_score=process_score.total,
                    total_updates=updates,
                    converged=convergence_detected,
                    pruned=final.pruned,
                )
            )
        return sorted(results, key=lambda item: item.trajectory_id)


__all__ = ["RecursiveRefinementController"]
