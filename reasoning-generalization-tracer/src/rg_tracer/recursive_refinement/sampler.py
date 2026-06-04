"""Self-play sampler for experimental recursive refinement."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ..concepts import trace_model
from ..trm_baseline import TinyRecursionModel
from .config import RecursiveRefinementConfig
from .controller import RecursiveRefinementController
from .logging import summarize_run, summarize_trajectory
from .projection import LatticeDeductionProjector
from .scoring import score_trajectory
from .types import LatticeDiagnostics, RefinementRun, RefinementState, TrajectoryResult


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def _initial_prediction(problem: Mapping[str, object]) -> object | None:
    numbers = problem.get("numbers", [])
    if isinstance(numbers, (list, tuple)) and numbers:
        ints = [item for item in numbers if isinstance(item, int) and not isinstance(item, bool)]
        if ints:
            return sum(ints)
    sequence = problem.get("sequence", [])
    if problem.get("task") == "parity" and isinstance(sequence, (list, tuple)):
        ints = [item for item in sequence if isinstance(item, int) and not isinstance(item, bool)]
        if ints:
            return sum(ints) % 2
    domain = problem.get("domain", [])
    if problem.get("task") == "finite_domain_constraint" and isinstance(domain, list) and domain:
        return domain[0]
    return 0


def _candidate_metrics(
    problem: Mapping[str, object],
    *,
    prediction: object | None,
    steps: int,
    contradictions: int = 0,
    lattice: LatticeDiagnostics | None = None,
) -> dict[str, Mapping[str, Any]]:
    expected = problem.get("answer")
    formal_proof = contradictions == 0
    if lattice is not None and lattice.mode in {"advisory", "gated", "shadow"}:
        contradictions += int(lattice.contradiction_detected)
        formal_proof = formal_proof and (lattice.resolved or lattice.mode == "shadow")
    error_rate = 0.0 if expected is None or prediction == expected else 0.5
    return {
        "logical_validity": {
            "formal_proof": formal_proof,
            "contradictions": contradictions,
        },
        "conceptual_clarity": {"undefined_symbols": 0, "jargon_flags": 0},
        "completeness": {
            "edge_cases": 0 if formal_proof else 1,
            "coverage_ratio": 1.0 if formal_proof else 0.6,
        },
        "rigor": {"checked_steps": max(1, steps), "total_steps": max(1, steps)},
        "efficiency": {"steps": max(1, steps), "baseline_steps": max(1, steps)},
        "heuristic_creativity": {"distinct_paths": 1, "novelty": 0.1},
        "numerical_accuracy": {"error_rate": error_rate, "error_tolerance": 0.1},
        "cognitive_efficiency": {
            "token_count": 18 + steps,
            "token_budget": 256,
            "time_ms": max(1, steps),
            "time_budget_ms": 64,
        },
        "explanatory_power": {"causal_links": 1, "examples": int(formal_proof)},
        "self_consistency": {"self_corrections": contradictions, "agreement_rate": 1.0},
        "abstraction_generalization": {
            "transfer_accuracy": 0.9,
            "compression_gain": 0.1,
            "variable_lifts": 1,
        },
    }


def _diagnostics_from_projection(result: object) -> LatticeDiagnostics:
    payload = result.as_dict() if hasattr(result, "as_dict") else {}
    steps = payload.get("projection_steps", [])
    if not isinstance(steps, list):
        steps = []
    return LatticeDiagnostics(
        mode=str(payload.get("mode", "off")),
        adapter_name=(
            str(payload.get("adapter_name")) if payload.get("adapter_name") is not None else None
        ),
        initial_candidate_count=len(payload.get("initial_candidates", []) or []),
        remaining_candidate_count=len(payload.get("remaining_candidates", []) or []),
        meet_count=int(payload.get("meet_count", 0)),
        join_count=int(payload.get("join_count", 0)),
        projection_count=len(steps),
        canonicalization_count=int(payload.get("canonicalization_count", 0)),
        merged_branch_count=int(payload.get("merged_branch_count", 0)),
        contradiction_detected=bool(payload.get("contradiction_detected", False)),
        resolved=bool(payload.get("resolved", False)),
        unresolved=bool(payload.get("unresolved", False)),
        abstain_recommended=bool(payload.get("abstain_recommended", False)),
        abstain_reason=(
            str(payload.get("abstain_reason"))
            if payload.get("abstain_reason") is not None
            else None
        ),
        projection_steps=[dict(step) for step in steps if isinstance(step, dict)],
    )


def _resolved_candidate(lattice: LatticeDiagnostics | None) -> object | None:
    if lattice is None or not lattice.resolved or len(lattice.projection_steps) == 0:
        return None
    final_step = lattice.projection_steps[-1]
    remaining = final_step.get("after_candidates")
    if isinstance(remaining, list) and len(remaining) == 1:
        return remaining[0]
    return None


class ToyTRMSampler:
    """Deterministic toy TRM-compatible sampler used by the ladder baselines."""

    sampler_name = "trm"

    def __init__(self, config: RecursiveRefinementConfig | None = None) -> None:
        self.config = config or RecursiveRefinementConfig()
        self.model = TinyRecursionModel()

    def generate(self, problem: Mapping[str, object], k: int) -> list[dict[str, object]]:
        candidates = []
        prediction = _initial_prediction(problem)
        for idx in range(k):
            confidence = min(0.95, 0.80 - 0.05 * idx)
            candidates.append(
                self._candidate(
                    problem,
                    prediction=prediction,
                    confidence=confidence,
                    steps=1,
                    trajectory_id=f"trm_{idx}",
                    text=f"Therefore the deterministic TRM prediction is {prediction}.",
                )
            )
        return candidates

    def _candidate(
        self,
        problem: Mapping[str, object],
        *,
        prediction: object | None,
        confidence: float,
        steps: int,
        trajectory_id: str,
        text: str,
        lattice: LatticeDiagnostics | None = None,
        refinement_run: dict[str, object] | None = None,
        perturbations: list[dict[str, object]] | None = None,
        process_score: float | None = None,
    ) -> dict[str, object]:
        trace = trace_model(
            self.sampler_name,
            str(problem.get("id", "unknown")),
            metadata={"trajectory_id": trajectory_id},
        ).to_json()
        if lattice is not None:
            trace["lattice"] = lattice.as_dict()
        if perturbations:
            trace["perturbations"] = perturbations
        score = process_score if process_score is not None else confidence
        return {
            "text": text,
            "confidence": confidence,
            "metrics": _candidate_metrics(
                problem, prediction=prediction, steps=steps, lattice=lattice
            ),
            "trace": trace,
            "prediction": prediction,
            "trajectory_id": trajectory_id,
            "process_score": score,
            "process_score_components": {"heuristic_confidence": confidence, "total": score},
            "total_updates": steps,
            "max_depth": steps,
            "max_width": 1,
            "converged": True,
            "budget_exhausted": False,
            "lattice_diagnostics": lattice.as_dict() if lattice else None,
            "perturbations": perturbations or [],
            "refinement_run": refinement_run,
        }


class PTRMSampler(ToyTRMSampler):
    """PTRM-inspired bounded Gaussian trajectory exploration over the toy TRM backend."""

    sampler_name = "ptrm"

    def generate(self, problem: Mapping[str, object], k: int) -> list[dict[str, object]]:
        from .perturbation import GaussianTrajectoryPerturber

        config = self.config.perturbation
        config.enabled = True
        width = max(1, min(k, config.trajectories))
        perturber = GaussianTrajectoryPerturber(config)
        results: list[TrajectoryResult] = []
        all_records = []
        for index in range(width):
            trajectory_id = f"ptrm_{index}"
            prediction = _initial_prediction(problem)
            confidence = _clamp_probability(0.72 - 0.03 * index)
            uncertainty = _clamp_probability(1.0 - confidence)
            state = RefinementState(
                trajectory_id=trajectory_id,
                parent_id=None,
                depth=0,
                state_value=confidence - uncertainty,
                uncertainty=uncertainty,
                confidence=confidence,
                prediction=prediction,
                active_views=[],
                operations=[],
            )
            states = [state]
            records = []
            for depth in range(1, self.config.budget.max_depth + 1):
                noise, record = perturber.maybe_perturb(
                    state,
                    trajectory_id=trajectory_id,
                    parent_id=None,
                    depth=depth,
                )
                next_prediction = self._shift_prediction(prediction, noise, index)
                confidence = _clamp_probability(state.confidence + 0.035 + noise * 0.25)
                uncertainty = _clamp_probability(state.uncertainty - 0.025 - noise * 0.15)
                state = replace(
                    state,
                    depth=depth,
                    state_value=confidence - uncertainty,
                    confidence=confidence,
                    uncertainty=uncertainty,
                    prediction=next_prediction,
                    perturbation=record,
                )
                states.append(state)
                if record is not None:
                    records.append(record)
                    all_records.append(record)
            process_score = state.confidence - state.uncertainty + 0.02 * len(records)
            results.append(
                TrajectoryResult(
                    trajectory_id=trajectory_id,
                    states=states,
                    prediction=state.prediction,
                    confidence=state.confidence,
                    process_score=process_score,
                    total_updates=max(0, len(states) - 1),
                    converged=False,
                    pruned=False,
                    process_score_components={
                        "confidence": state.confidence,
                        "uncertainty": state.uncertainty,
                        "perturbation_count": len(records),
                        "total": process_score,
                    },
                    perturbations=records,
                )
            )
        ranked = sorted(results, key=lambda item: item.process_score, reverse=True)
        run = RefinementRun(
            trajectories=ranked,
            selected_trajectory_id=ranked[0].trajectory_id,
            total_updates=sum(item.total_updates for item in ranked),
            max_observed_depth=max(item.total_updates for item in ranked),
            max_observed_width=width,
            convergence_detected=False,
            budget_exhausted=False,
            perturbations=all_records,
        )
        return [self._candidate_from_result(problem, run, item) for item in ranked[:k]]

    def _candidate_from_result(
        self,
        problem: Mapping[str, object],
        run: RefinementRun,
        trajectory: TrajectoryResult,
    ) -> dict[str, object]:
        perturbations = [record.as_dict() for record in trajectory.perturbations]
        return self._candidate(
            problem,
            prediction=trajectory.prediction,
            confidence=trajectory.confidence,
            steps=trajectory.total_updates,
            trajectory_id=trajectory.trajectory_id,
            text=(
                f"Therefore the PTRM-inspired widened prediction is {trajectory.prediction} "
                f"because bounded trajectories were scored by public confidence."
            ),
            refinement_run=run.as_dict(),
            perturbations=perturbations,
            process_score=trajectory.process_score,
        ) | {
            "trajectory_metadata": summarize_trajectory(trajectory),
            "max_width": run.max_observed_width,
            "refinement_summary": summarize_run(run),
        }

    def _shift_prediction(
        self,
        prediction: object | None,
        noise: float,
        trajectory_index: int,
    ) -> object | None:
        if isinstance(prediction, int) and not isinstance(prediction, bool):
            if abs(noise) < 0.12:
                return prediction
            direction = 1 if noise > 0 else -1
            return prediction + direction * ((trajectory_index % 2) + 1)
        return prediction


class LatticeTRMSampler(ToyTRMSampler):
    """Deterministic TRM with explicit task-local LDT-inspired lattice projection."""

    sampler_name = "lattice_trm"

    def generate(self, problem: Mapping[str, object], k: int) -> list[dict[str, object]]:
        projector = LatticeDeductionProjector(self.config.lattice)
        lattice = _diagnostics_from_projection(projector.project(problem))
        prediction = _initial_prediction(problem)
        resolved = _resolved_candidate(lattice)
        if self.config.lattice.mode == "gated" and resolved is not None:
            prediction = resolved
        confidence = 0.86 if not lattice.abstain_recommended else 0.20
        text = f"Therefore lattice-projected TRM prediction is {prediction}."
        if self.config.lattice.mode == "gated" and lattice.abstain_recommended:
            text = (
                f"I don't know. Lattice projection recommends abstention: {lattice.abstain_reason}."
            )
        candidate = self._candidate(
            problem,
            prediction=prediction if not lattice.abstain_recommended else None,
            confidence=confidence,
            steps=max(1, lattice.projection_count),
            trajectory_id="lattice_trm_0",
            text=text,
            lattice=lattice,
        )
        return [candidate for _ in range(max(1, k))]


class LatticePTRMSampler(PTRMSampler):
    """PTRM-inspired exploration constrained by explicit task-local lattice projection."""

    sampler_name = "lattice_ptrm"

    def generate(self, problem: Mapping[str, object], k: int) -> list[dict[str, object]]:
        widened = super().generate(problem, k)
        projector = LatticeDeductionProjector(self.config.lattice)
        branches = []
        seen: set[tuple[object, ...]] = set()
        merged = 0
        pruned = 0
        for raw in widened:
            lattice = _diagnostics_from_projection(projector.project(problem))
            resolved = _resolved_candidate(lattice)
            key = tuple(
                tuple(step.get("after_candidates", []))
                for step in lattice.projection_steps[-1:]
                if isinstance(step, dict)
            )
            if self.config.lattice.merge_equivalent_branches and key in seen:
                merged += 1
                continue
            seen.add(key)
            if lattice.contradiction_detected:
                pruned += 1
                continue
            prediction = resolved if resolved is not None else raw.get("prediction")
            if self.config.lattice.mode == "gated" and lattice.abstain_recommended:
                raw["text"] = f"I don't know. Lattice branch abstained: {lattice.abstain_reason}."
                raw["confidence"] = 0.20
                raw["prediction"] = None
            elif self.config.lattice.mode == "gated" and resolved is not None:
                raw["prediction"] = prediction
                raw["text"] = f"Therefore lattice-constrained PTRM prediction is {prediction}."
                raw["confidence"] = max(float(raw.get("confidence", 0.0)), 0.86)
            lattice.merged_branch_count = merged
            raw["lattice_diagnostics"] = lattice.as_dict()
            raw["metrics"] = _candidate_metrics(
                problem,
                prediction=raw.get("prediction"),
                steps=int(raw.get("total_updates", 1)),
                lattice=lattice,
            )
            trace = raw.get("trace")
            if isinstance(trace, dict):
                trace["lattice"] = lattice.as_dict()
            branches.append(raw)
        if not branches:
            lattice = _diagnostics_from_projection(projector.project(problem))
            lattice.abstain_recommended = True
            lattice.abstain_reason = lattice.abstain_reason or "no_surviving_branch"
            branches.append(
                self._candidate(
                    problem,
                    prediction=None,
                    confidence=0.20,
                    steps=1,
                    trajectory_id="lattice_ptrm_abstain",
                    text="I don't know. No surviving safe lattice-constrained branch resolved.",
                    lattice=lattice,
                )
            )
        for branch in branches:
            diagnostics = branch.get("lattice_diagnostics")
            if isinstance(diagnostics, dict):
                diagnostics["pruned_branch_count"] = pruned
                diagnostics["merged_branch_count"] = merged
        return sorted(
            branches,
            key=lambda item: (
                bool((item.get("lattice_diagnostics") or {}).get("resolved")),
                float(item.get("confidence", 0.0)),
            ),
            reverse=True,
        )[:k]


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


__all__ = [
    "GRAMMDTSampler",
    "LatticePTRMSampler",
    "LatticeTRMSampler",
    "PTRMSampler",
    "ToyTRMSampler",
]
