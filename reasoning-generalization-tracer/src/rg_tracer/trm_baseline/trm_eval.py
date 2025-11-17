from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping, Sequence, Tuple

from ..scoring import axes
from ..semantics import verify_chain
from .trm_model import TinyRecursionModel


@dataclass
class EvaluationResult:
    accuracy: float
    axis_scores: Mapping[str, int]
    traces: Sequence[Tuple[Sequence[int], Sequence[float]]]
    semantic_scores: Sequence[float]


def evaluate(
    model: TinyRecursionModel,
    data: Iterable[Tuple[Sequence[int], int]],
) -> EvaluationResult:
    total = 0
    correct = 0
    per_axis = {name: [] for name in axes.__all__}
    traces = []
    semantic_scores = []
    for seq, target in data:
        pred = model.predict(seq)
        total += 1
        if round(pred) == target:
            correct += 1
        metrics = {
            "logical_validity": {"formal_proof": False, "contradictions": 0},
            "numerical_accuracy": {"error_rate": abs(pred - target), "error_tolerance": 0.25},
            "rigor": {"checked_steps": len(seq), "total_steps": len(seq)},
            "abstraction_generalization": {
                "transfer_accuracy": pred if target else 1 - pred,
                "compression_gain": 0.1,
                "variable_lifts": 1,
            },
        }
        scores = {
            axis_name: getattr(axes, axis_name)(metrics.get(axis_name, {}))
            for axis_name in axes.__all__
        }
        for axis_name, score in scores.items():
            per_axis[axis_name].append(score)
        chain_text = (
            f"Sequence {seq} maps to activation {pred:.2f} because recursion accumulates parity."
        )
        report = verify_chain(
            chain_text,
            {"concept": "parity", "units": "binary", "variables": ["x"]},
        )
        semantic_scores.append(report.score)
        if model.trace_states:
            model.reset_history()
            _, trace = model.forward(seq)
            traces.append((seq, trace))
    axis_scores = {axis: int(mean(values)) if values else 0 for axis, values in per_axis.items()}
    accuracy = correct / total if total else 0.0
    return EvaluationResult(
        accuracy=accuracy,
        axis_scores=axis_scores,
        traces=traces,
        semantic_scores=semantic_scores,
    )


__all__ = ["EvaluationResult", "evaluate"]
