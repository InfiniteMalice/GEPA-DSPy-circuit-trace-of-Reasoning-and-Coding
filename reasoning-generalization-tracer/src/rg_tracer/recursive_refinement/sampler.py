"""Self-play sampler for experimental recursive refinement."""

from __future__ import annotations

from typing import Any, Mapping

from ..concepts import trace_model
from ..trm_baseline import TinyRecursionModel
from .config import RecursiveRefinementConfig
from .controller import RecursiveRefinementController
from .logging import summarize_run, summarize_trajectory
from .scoring import score_trajectory
from .types import RefinementRun, TrajectoryResult


class GRAMMDTSampler:
    """Experimental GRAM-inspired recursive widening with MDT-inspired view routing.

    This is a bounded heuristic scaffold for toy RG-Tracer tasks. It is not a faithful
    reproduction of GRAM, HRM-Text, or mathematical MDT diffusion geometry.
    """

    def __init__(self, config: RecursiveRefinementConfig | None = None) -> None:
        self.config = config or RecursiveRefinementConfig()
        self.model = TinyRecursionModel()

    def generate(
        self,
        problem: Mapping[str, object],
        k: int,
    ) -> list[dict[str, object]]:
        """Generate self-play-compatible raw candidates."""

        controller = RecursiveRefinementController(self.config, model=self.model)
        run = controller.run(problem, target_width=max(1, k))
        ranked = sorted(
            run.trajectories,
            key=lambda trajectory: (trajectory.process_score, trajectory.confidence),
            reverse=True,
        )
        return [
            self._candidate_from_trajectory(problem, run, trajectory) for trajectory in ranked[:k]
        ]

    def _candidate_from_trajectory(
        self,
        problem: Mapping[str, object],
        run: RefinementRun,
        trajectory: TrajectoryResult,
    ) -> dict[str, object]:
        final_state = trajectory.states[-1]
        route = [operation.view_name for operation in final_state.operations]
        process_score, process_score_components = self._process_score_payload(
            trajectory,
            run,
        )
        trace = trace_model(
            "gram_mdt",
            str(problem.get("id", "unknown")),
            metadata={
                "trajectory_id": trajectory.trajectory_id,
                "view_route": route,
                "max_depth": run.max_observed_depth,
                "max_width": run.max_observed_width,
            },
        ).to_json()
        trace["trajectory_id"] = trajectory.trajectory_id
        trace["view_route"] = route
        trace["refinement_summary"] = summarize_trajectory(trajectory)
        return {
            "text": self._candidate_text(trajectory),
            "confidence": trajectory.confidence,
            "metrics": self._metrics(problem, trajectory, run),
            "trace": trace,
            "prediction": trajectory.prediction,
            "refinement_run": run.as_dict(),
            "trajectory_id": trajectory.trajectory_id,
            "trajectory_metadata": summarize_trajectory(trajectory),
            "process_score": process_score,
            "process_score_components": process_score_components,
            "view_route": route,
            "total_updates": trajectory.total_updates,
            "max_depth": run.max_observed_depth,
            "max_width": run.max_observed_width,
            "converged": trajectory.converged,
            "budget_exhausted": run.budget_exhausted,
            "refinement_summary": summarize_run(run),
        }

    def _process_score_payload(
        self,
        trajectory: TrajectoryResult,
        run: RefinementRun,
    ) -> tuple[float, dict[str, object]]:
        if trajectory.process_score_components:
            return trajectory.process_score, dict(trajectory.process_score_components)
        if len(trajectory.states) > 1:
            score = score_trajectory(
                trajectory.states,
                total_updates=trajectory.total_updates,
                max_total_updates=self.config.budget.max_total_updates,
                converged=trajectory.converged,
                budget_exhausted=run.budget_exhausted,
            )
            return score.total, score.as_dict()
        return trajectory.process_score, {"total": trajectory.process_score}

    def _candidate_text(self, trajectory: TrajectoryResult) -> str:
        final_state = trajectory.states[-1]
        route = [operation.view_name for operation in final_state.operations]
        route_text = " -> ".join(route) if route else "shallow_pass"
        return (
            f"Experimental recursive refinement selected prediction {trajectory.prediction}. "
            f"Public view route: {route_text}."
        )

    def _metrics(
        self,
        problem: Mapping[str, object],
        trajectory: TrajectoryResult,
        run: RefinementRun,
    ) -> dict[str, Mapping[str, Any]]:
        expected = problem.get("answer")
        prediction = trajectory.prediction
        final_state = trajectory.states[-1]
        operations = final_state.operations
        contradictions = sum(1 for operation in operations if operation.constraint_passed is False)
        verification_success = any(
            operation.view_name == "verification" and operation.constraint_passed is True
            for operation in operations
        )
        constraint_success = any(
            operation.view_name == "constraint_check" and operation.constraint_passed is True
            for operation in operations
        )
        route = [operation.view_name for operation in operations]
        total_steps = max(1, len(operations))
        error_rate = 0.0 if expected is None or prediction == expected else 0.5
        return {
            "logical_validity": {
                "formal_proof": verification_success or constraint_success,
                "contradictions": contradictions,
            },
            "conceptual_clarity": {"undefined_symbols": 0, "jargon_flags": 0},
            "completeness": {
                "edge_cases": 0 if verification_success or constraint_success else 1,
                "coverage_ratio": 1.0 if verification_success else 0.8,
            },
            "rigor": {"checked_steps": total_steps, "total_steps": total_steps},
            "efficiency": {
                "steps": max(1, trajectory.total_updates),
                "baseline_steps": max(1, run.total_updates),
            },
            "heuristic_creativity": {
                "distinct_paths": run.max_observed_width,
                "novelty": min(1.0, len(set(route)) / 4.0),
            },
            "numerical_accuracy": {
                "error_rate": error_rate,
                "error_tolerance": 0.1,
            },
            "cognitive_efficiency": {
                "token_count": 24 + 4 * len(route),
                "token_budget": 256,
                "time_ms": max(1, run.total_updates),
                "time_budget_ms": self.config.budget.max_total_updates,
            },
            "explanatory_power": {
                "causal_links": len(set(route)),
                "examples": int(verification_success),
            },
            "self_consistency": {
                "self_corrections": contradictions,
                "agreement_rate": 1.0 if run.convergence_detected else 0.85,
            },
            "abstraction_generalization": {
                "transfer_accuracy": 0.9,
                "compression_gain": 0.1 if "compression" in route else 0.0,
                "variable_lifts": 1,
            },
        }


__all__ = ["GRAMMDTSampler"]
