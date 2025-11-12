"""Self-play runner producing Pareto-selected reasoning chains."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping as MappingABC
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

from ..abstention import apply_abstention
from ..attribution import graphs as attr_graphs
from ..attribution import metrics as attr_metrics
from ..concepts import ConceptSpec, compute_concept_reward, trace_model
from ..scoring import aggregator, axes
from ..semantics import SemanticTag, repair_once, verify_chain
from ..semantics.patterns import build_token_boundary_pattern
from ..trm_baseline import TinyRecursionModel

AXIS_FUNCTIONS = {name: getattr(axes, name) for name in axes.__all__}
ATTR_PHASES = ["pre_overfit", "overfit", "pre_grok", "post_grok"]
DEFAULT_ATTR_CONFIG = {
    "probe_size": len(ATTR_PHASES),
    "topk": 2,
    "backend": "null",
}
DEFAULT_ATTR_BONUSES = {
    "alignment_gain": 0.01,
    "repeatability_gain": 0.01,
    "sparsity_drop": 0.005,
}


@dataclass
class Candidate:
    text: str
    confidence: float
    metrics: Dict[str, Mapping[str, object]]
    axis_scores: Dict[str, int]
    composite: float
    base_composite: float
    passes_gates: bool
    failed_gates: Dict[str, float]
    concept_reward: float
    abstained: bool
    trace: Dict[str, object] | None
    problem_id: str
    semantic_report: Mapping[str, object]
    semantics_map: Dict[str, List[str]] = field(default_factory=dict)
    attr_metrics: Dict[str, float] | None = None
    attr_bonus: float = 0.0


class TRMSampler:
    """Lightweight sampler using the Tiny Recursion Model for toy tasks."""

    def __init__(self) -> None:
        self.model = TinyRecursionModel()

    def generate(self, problem: Mapping[str, object], k: int) -> List[Dict[str, object]]:
        numbers = problem.get("numbers", [])
        answer = problem.get("answer")
        parity_task = problem.get("task") == "parity"
        base_confidence = 0.8 if answer is not None else 0.6
        candidates: List[Dict[str, object]] = []
        for idx in range(k):
            if numbers:
                prediction = sum(numbers)
            elif parity_task:
                seq = problem.get("sequence", [0, 1])
                prediction = sum(seq) % 2
            else:
                prediction = 0
            confidence = min(0.95, base_confidence - 0.05 * idx)
            metrics = {
                "logical_validity": {"formal_proof": True, "contradictions": 0},
                "numerical_accuracy": {
                    "error_rate": 0.0 if prediction == answer else 0.5,
                    "error_tolerance": 0.1,
                },
                "rigor": {
                    "checked_steps": len(numbers) or 1,
                    "total_steps": len(numbers) or 1,
                },
                "efficiency": {
                    "steps": len(numbers) or 1,
                    "baseline_steps": max(len(numbers), 1),
                },
                "abstraction_generalization": {
                    "transfer_accuracy": 0.9,
                    "compression_gain": 0.1,
                    "variable_lifts": 1,
                },
            }
            steps = [f"Add {numbers} to get {prediction}."]
            text = "\n".join(steps)
            trace = trace_model("trm", str(problem.get("id", "unknown"))).to_json()
            candidates.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "metrics": metrics,
                    "trace": trace,
                }
            )
        return candidates


def _compute_axis_scores(metrics: Mapping[str, Mapping[str, object]]) -> Dict[str, int]:
    scores = {}
    for axis_name, func in AXIS_FUNCTIONS.items():
        axis_metrics = metrics.get(axis_name, {})
        scores[axis_name] = func(axis_metrics)
    return scores


def _dominates(a: Mapping[str, int], b: Mapping[str, int]) -> bool:
    ge_all = all(a.get(axis, 0) >= b.get(axis, 0) for axis in AXIS_FUNCTIONS)
    gt_any = any(a.get(axis, 0) > b.get(axis, 0) for axis in AXIS_FUNCTIONS)
    return ge_all and gt_any


def pareto_frontier(candidates: Sequence[Candidate]) -> List[Candidate]:
    frontier: List[Candidate] = []
    for candidate in candidates:
        if any(
            _dominates(other.axis_scores, candidate.axis_scores)
            for other in candidates
            if other is not candidate
        ):
            continue
        frontier.append(candidate)
    return frontier


def _prepare_output_dir(base_dir: str | Path | None = None) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = Path(base_dir or Path.cwd() / "runs") / timestamp
    base.mkdir(parents=True, exist_ok=True)
    return base


def _build_probe_inputs(
    problem: Mapping[str, object],
    probe_size: int,
) -> List[Dict[str, object]]:
    if probe_size <= 0:
        return []
    sequence = problem.get("sequence")
    numbers = problem.get("numbers")

    def _as_list(value: object) -> List[object]:
        if isinstance(value, (list, tuple)):
            return list(value)
        return []

    base_tokens = _as_list(sequence)
    if not base_tokens:
        base_tokens = _as_list(numbers)
    if not base_tokens:
        prompt = str(problem.get("prompt", "")) or str(problem.get("statement", ""))
        # Placeholder IDs keep BackendNull deterministic; other backends warn upstream.
        base_tokens = [ord(ch) % 10 for ch in prompt[:8]] or [0]
    task_id = str(problem.get("id", "task"))
    probes: List[Dict[str, object]] = []
    for index in range(probe_size):
        phase = ATTR_PHASES[index % len(ATTR_PHASES)]
        probes.append(
            {
                "task_id": f"{task_id}_probe_{index}",
                "tokens": list(base_tokens),
                "phase": phase,
                "probe_index": index,
            }
        )
    return probes


def _concept_feature_descriptors(concept: ConceptSpec | None) -> List[Mapping[str, object]]:
    """Return concept feature descriptors suitable for attribution metrics."""

    if concept is None:
        return []
    if concept.feature_catalog:
        descriptors: List[Mapping[str, object]] = []
        for entry in concept.feature_catalog:
            if not isinstance(entry, Mapping):
                continue
            identifier = entry.get("id")
            if identifier is None:
                continue
            descriptor: Dict[str, object] = {"id": str(identifier)}
            tags = entry.get("tags")
            if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)):
                descriptor["tags"] = [str(tag) for tag in tags]
            for key in ("layer", "type"):
                if key in entry:
                    descriptor[key] = entry[key]
            descriptors.append(descriptor)
        if descriptors:
            return descriptors
    fallback: List[Mapping[str, object]] = []
    for item in concept.expected_substructures:
        if not item:
            continue
        fallback.append({"id": str(item), "tags": [str(item)]})
    return fallback


def _resolve_attr_param(
    config: Mapping[str, object],
    key: str,
    default: int,
    *,
    coerce: Callable[[object], int] = int,
) -> int:
    raw_value = config.get(key)
    return default if raw_value is None else coerce(raw_value)


def _compute_and_apply_attr_metrics(
    candidate: Candidate,
    graphs: List[Mapping[str, object]],
    bonuses: Mapping[str, float],
    concept: ConceptSpec | None,
) -> Dict[str, float]:
    """Compute attribution metrics and update the candidate-side bookkeeping."""
    concept_features = _concept_feature_descriptors(concept)
    metrics = {
        "sparsity": attr_metrics.path_sparsity(graphs),
        "avg_path_length": attr_metrics.average_path_length(graphs),
        "branching_factor": attr_metrics.average_branching_factor(graphs),
        "repeatability": attr_metrics.repeatability(graphs),
        "alignment": attr_metrics.concept_alignment(graphs, concept_features),
        "delta_sparsity": attr_metrics.delta_sparsity(graphs),
        "delta_alignment": attr_metrics.delta_alignment(graphs, concept_features),
        "delta_repeatability": attr_metrics.delta_repeatability(graphs),
    }
    bonus = 0.0
    if metrics["delta_alignment"] > 0:
        bonus += bonuses.get("alignment_gain", 0.0)
    if metrics["delta_repeatability"] > 0:
        bonus += bonuses.get("repeatability_gain", 0.0)
    if metrics["delta_sparsity"] > 0:
        bonus += bonuses.get("sparsity_drop", 0.0)
    candidate.attr_metrics = metrics
    candidate.attr_bonus = bonus
    candidate.composite = candidate.base_composite + candidate.attr_bonus
    return {**metrics, "bonus": bonus}


def _apply_attribution_rewards(
    candidates: List[Candidate],
    *,
    model: object,
    problem: Mapping[str, object],
    run_dir: Path,
    profile_bonuses: Mapping[str, float],
    attr_config: Mapping[str, object],
    concept: ConceptSpec | None,
) -> None:
    if not candidates:
        return
    probe_size = _resolve_attr_param(
        attr_config,
        "probe_size",
        DEFAULT_ATTR_CONFIG["probe_size"],
    )
    topk = _resolve_attr_param(
        attr_config,
        "topk",
        DEFAULT_ATTR_CONFIG["topk"],
    )
    backend_value = attr_config.get("backend", DEFAULT_ATTR_CONFIG["backend"])
    backend_name = str(backend_value or DEFAULT_ATTR_CONFIG["backend"]).strip().lower()
    if not backend_name:
        backend_name = DEFAULT_ATTR_CONFIG["backend"]
    if probe_size <= 0 or topk <= 0:
        return
    has_sequence = isinstance(problem.get("sequence"), (list, tuple))
    has_numbers = isinstance(problem.get("numbers"), (list, tuple))
    has_explicit_tokens = has_sequence or has_numbers
    if backend_name != "null" and not has_explicit_tokens:
        warnings.warn(
            "Non-null attribution backend configured without explicit tokens; "
            "provide vocabulary-aligned probes to avoid placeholder IDs.",
            RuntimeWarning,
            stacklevel=2,
        )
    probes = _build_probe_inputs(problem, probe_size)
    if not probes:
        return
    attr_dir = run_dir / "attr"
    attr_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "attr_metrics.jsonl"
    try:
        backend = attr_graphs.get_backend(backend_name)
    except KeyError as exc:  # pragma: no cover - configuration error
        warnings.warn(
            f"Unknown attribution backend '{backend_name}': {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    except Exception as exc:  # pragma: no cover - backend initialisation failure
        warnings.warn(
            f"Failed to initialise attribution backend '{backend_name}': {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    bonuses = {**DEFAULT_ATTR_BONUSES, **dict(profile_bonuses)}
    top_pairs = sorted(
        enumerate(candidates),
        key=lambda item: item[1].composite,
        reverse=True,
    )[:topk]
    with metrics_path.open("w", encoding="utf8") as handle:
        for idx, candidate in top_pairs:
            graphs: List[Mapping[str, object]] = []
            for probe in probes:
                payload = dict(probe)
                seed = idx * 997 + int(payload.get("probe_index", 0))
                graph = attr_graphs.extract_graph(
                    model,
                    payload,
                    backend=backend,
                    seed=seed,
                )
                graphs.append(graph)
                attr_path = attr_dir / f"candidate{idx}_probe{payload['probe_index']}.json"
                with attr_path.open("w", encoding="utf8") as graph_handle:
                    json.dump(graph, graph_handle, indent=2)
            metrics = _compute_and_apply_attr_metrics(candidate, graphs, bonuses, concept)
            record = {
                "candidate_index": idx,
                "problem_id": candidate.problem_id,
                **metrics,
            }
            handle.write(json.dumps(record) + "\n")


def _load_problem(path: str | Path) -> Mapping[str, object]:
    with open(path, "r", encoding="utf8") as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError("Problem file is empty")
        return json.loads(first_line)


def _map_semantics_to_features(
    trace: Mapping[str, object],
    report: Mapping[str, object],
) -> Dict[str, List[str]]:
    features = trace.get("features", []) if trace else []
    entailed_steps = [
        entry.get("step", "")
        for entry in report.get("tags", [])
        if SemanticTag.ENTAILED.value in entry.get("tags", [])
    ]
    contradictory_steps = [
        entry.get("step", "")
        for entry in report.get("tags", [])
        if SemanticTag.CONTRADICTION.value in entry.get("tags", [])
    ]
    entailed_lower = [str(step).lower() for step in entailed_steps if step is not None]
    contradictory_lower = [str(step).lower() for step in contradictory_steps if step is not None]
    entailed_ids: List[str] = []
    contradictory_ids: List[str] = []
    for feature in features:
        raw_tags = feature.get("tags")
        # Handle tags sourced from different pipelines (None, scalar, iterable, or blank strings).
        if raw_tags is None:
            tags_iter = []
        elif isinstance(raw_tags, (list, tuple, set)):
            tags_iter = raw_tags
        else:
            tags_iter = [raw_tags]
        cleaned_tags: List[str] = []
        for tag in tags_iter:
            text_tag = str(tag).strip()
            if text_tag:
                cleaned_tags.append(text_tag.lower())
        lower_tags = set(cleaned_tags)
        feature_id = str(feature.get("id", ""))
        entailed_match = False
        contradictory_match = False
        for tag in lower_tags:
            pattern = build_token_boundary_pattern(tag)
            if not pattern:
                continue
            if any(pattern.search(step) for step in entailed_lower):
                entailed_match = True
            if any(pattern.search(step) for step in contradictory_lower):
                contradictory_match = True
        if entailed_match:
            entailed_ids.append(feature_id)
        if contradictory_match:
            contradictory_ids.append(feature_id)
    return {
        "entailed_feature_ids": entailed_ids,
        "contradictory_feature_ids": contradictory_ids,
    }


def run_self_play(
    problem_path: str | Path,
    *,
    profile: str = "proof_math",
    k: int = 4,
    sampler: str = "trm",
    concept: ConceptSpec | None = None,
    output_dir: str | Path | None = None,
) -> Dict[str, object]:
    problem = _load_problem(problem_path)
    if sampler != "trm":
        raise ValueError(f"Unsupported sampler: {sampler}")
    sampler_impl = TRMSampler()
    raw_candidates = sampler_impl.generate(problem, k)

    profiles = aggregator.load_profiles()
    profile_config = aggregator.get_last_config()
    raw_attr_config = profile_config.get("attr") if profile_config else {}
    if isinstance(raw_attr_config, MappingABC):
        attr_config: Mapping[str, object] = dict(raw_attr_config)
    else:
        attr_config = {}
    if profile not in profiles:
        raise KeyError(f"Profile {profile} not found")
    profile_obj = profiles[profile]
    run_dir = _prepare_output_dir(output_dir)

    results: List[Candidate] = []
    semantics_logs: List[Dict[str, object]] = []
    problem_id = str(problem.get("id", "task"))
    expected_units = str(problem.get("units", "")) or None
    preferred_vars = problem.get("variables", [])

    for raw in raw_candidates:
        axis_scores = _compute_axis_scores(raw.get("metrics", {}))
        eval_result = aggregator.evaluate_profile(axis_scores, profile_obj)
        initial_text = raw.get("text", "")
        report = verify_chain(initial_text, problem)
        repairs_attempted = 0
        text_after_repair = initial_text
        if raw.get("confidence", 0.0) < 0.75 or report.score < 2:
            repairs_attempted = 1
            repaired_steps = repair_once(
                text_after_repair,
                report.tags,
                expected_units=expected_units,
                preferred_variables=preferred_vars,
            )
            text_after_repair = "\n".join(repaired_steps)
            report = verify_chain(text_after_repair, problem)
        gates_pass = bool(eval_result["passes_gates"])
        abstention = apply_abstention(
            text_after_repair,
            raw.get("confidence", 0.0),
            report.score,
            gates_pass,
        )
        semantic_dict = report.as_dict()
        semantic_dict["repairs_attempted"] = repairs_attempted
        semantic_dict["abstained"] = abstention.abstained
        semantics_logs.append(semantic_dict)

        trace_obj = raw.get("trace")
        trace_json = trace_obj.to_json() if hasattr(trace_obj, "to_json") else trace_obj
        semantics_map = _map_semantics_to_features(trace_json, semantic_dict) if trace_json else {}
        composite_value = float(eval_result["composite"])
        candidate = Candidate(
            text=abstention.text,
            confidence=raw.get("confidence", 0.0),
            metrics=raw.get("metrics", {}),
            axis_scores=axis_scores,
            composite=composite_value,
            base_composite=composite_value,
            passes_gates=gates_pass,
            failed_gates=dict(eval_result["failed_gates"]),
            concept_reward=0.0,
            abstained=abstention.abstained,
            trace=trace_json,
            problem_id=problem_id,
            semantic_report=semantic_dict,
            semantics_map=semantics_map,
        )
        results.append(candidate)

    _apply_attribution_rewards(
        results,
        model=sampler_impl.model,
        problem=problem,
        run_dir=run_dir,
        profile_bonuses=profile_obj.bonuses,
        attr_config=attr_config,
        concept=concept,
    )

    if concept is not None:
        for candidate in results:
            if not candidate.trace:
                continue
            task_metrics = {
                **candidate.semantics_map,
                "concept_reuse": 1.0,
                "supporting_tasks": 1.0,
            }
            alignment_value = None
            if candidate.attr_metrics:
                alignment_value = candidate.attr_metrics.get("alignment")
            candidate.concept_reward = compute_concept_reward(
                candidate.trace,
                concept,
                task_metrics=task_metrics,
                alignment=alignment_value,
            )
            candidate.composite = (
                candidate.base_composite + candidate.attr_bonus + candidate.concept_reward
            )
    else:
        for candidate in results:
            candidate.composite = candidate.base_composite + candidate.attr_bonus

    frontier = pareto_frontier(results)
    best = max(results, key=lambda c: c.composite)

    scores_path = run_dir / "scores.jsonl"
    with scores_path.open("w", encoding="utf8") as handle:
        for candidate in results:
            handle.write(
                json.dumps(
                    {
                        "text": candidate.text,
                        "confidence": candidate.confidence,
                        "axis_scores": candidate.axis_scores,
                        "base_composite": candidate.base_composite,
                        "composite": candidate.composite,
                        "passes_gates": candidate.passes_gates,
                        "failed_gates": candidate.failed_gates,
                        "concept_reward": candidate.concept_reward,
                        "abstained": candidate.abstained,
                        "problem_id": candidate.problem_id,
                        "attr_bonus": candidate.attr_bonus,
                        "attr_metrics": candidate.attr_metrics,
                    }
                )
                + "\n"
            )

    semantics_path = run_dir / "semantics.jsonl"
    with semantics_path.open("w", encoding="utf8") as handle:
        for entry in semantics_logs:
            handle.write(json.dumps(entry) + "\n")

    summary_path = run_dir / "summary.md"
    with summary_path.open("w", encoding="utf8") as handle:
        handle.write("| # | Composite | Gates | Concept | Abstained | Semantic Score | Repairs |\n")
        handle.write("| - | --------- | ----- | ------- | --------- | ------------- | ------- |\n")
        for idx, candidate in enumerate(results, start=1):
            summary_row = (
                "| {idx} | {comp:.3f} | {gates} | {reward:.3f} | {abst} | {sem} | " "{repair} |\n"
            ).format(
                idx=idx,
                comp=candidate.composite,
                gates=candidate.passes_gates,
                reward=candidate.concept_reward,
                abst=candidate.abstained,
                sem=candidate.semantic_report.get("score", 0),
                repair=candidate.semantic_report.get("repairs_attempted", 0),
            )
            handle.write(summary_row)
        metrics_to_write = [
            (idx, candidate)
            for idx, candidate in enumerate(results, start=1)
            if candidate.attr_metrics
        ]
        if metrics_to_write:
            handle.write("\n### Attribution Metrics\n")
            for idx, candidate in metrics_to_write:
                metrics = candidate.attr_metrics or {}
                attr_line = (
                    "- #{idx} Δalign={align:.3f}, Δrepeat={repeat:.3f}, "
                    "Δsparsity={sparsity:.3f}\n"
                ).format(
                    idx=idx,
                    align=metrics.get("delta_alignment", 0.0),
                    repeat=metrics.get("delta_repeatability", 0.0),
                    sparsity=metrics.get("delta_sparsity", 0.0),
                )
                handle.write(attr_line)

    best_path = run_dir / "best.json"
    with best_path.open("w", encoding="utf8") as handle:
        json.dump(
            {
                "text": best.text,
                "confidence": best.confidence,
                "axis_scores": best.axis_scores,
                "composite": best.composite,
                "passes_gates": best.passes_gates,
                "failed_gates": best.failed_gates,
                "concept_reward": best.concept_reward,
                "abstained": best.abstained,
                "problem_id": best.problem_id,
            },
            handle,
            indent=2,
        )

    return {
        "run_dir": str(run_dir),
        "candidates": results,
        "frontier": frontier,
        "best": best,
    }


__all__ = ["run_self_play", "pareto_frontier", "Candidate"]
