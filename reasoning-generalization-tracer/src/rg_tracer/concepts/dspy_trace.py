"""DSPy integration for generating circuit traces."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import dspy
from dspy import ChainOfThought as CoT
from dspy import Signature as Sig

from .circuit_adapter import CircuitEdge, CircuitFeature, CircuitTrace, trace_model


class TraceSig(Sig):
    """Signature describing the circuit trace request."""

    ref: str = dspy.InputField(desc="Model reference")
    task: str = dspy.InputField(desc="Task identifier")
    meta: Mapping[str, Any] | None = dspy.InputField(
        desc="Optional metadata",
        optional=True,
    )
    trace: Mapping[str, Any] = dspy.OutputField(desc="Normalised circuit trace")


class TraceModule(dspy.Module):
    """Module invoking the circuit tracer via a Chain-of-Thought."""

    def __init__(self) -> None:
        super().__init__()
        self.cot = CoT(TraceSig)

    def forward(
        self,
        ref: str,
        task: str,
        meta: Mapping[str, Any] | None = None,
    ) -> dspy.Prediction:
        """Produce a circuit trace for the given model and task."""
        prediction = self.cot(ref=ref, task=task, meta=meta)
        if prediction.trace:
            return prediction
        circuit = trace_model(ref, task, metadata=meta)
        return dspy.Prediction(trace=circuit.to_json())


def _build_features(data: Iterable[Mapping[str, Any]]) -> list[CircuitFeature]:
    return [
        CircuitFeature(
            id=str(feature.get("id")),
            layer=int(feature.get("layer", 0)),
            importance=float(feature.get("importance", 0.0)),
            tags=list(feature.get("tags", [])),
        )
        for feature in data
    ]


def _build_edges(data: Iterable[Mapping[str, Any]]) -> list[CircuitEdge]:
    return [
        CircuitEdge(
            src=str(edge.get("src")),
            dst=str(edge.get("dst")),
            weight=float(edge.get("weight", 0.0)),
        )
        for edge in data
    ]


class TraceRunner:
    """Helper wrapping :class:`TraceModule` for repeated tracing."""

    def __init__(self) -> None:
        self.module = TraceModule()

    def run(
        self,
        model_ref: str,
        task_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> CircuitTrace:
        """Return a :class:`CircuitTrace` produced by the DSPy module."""
        result = self.module(ref=model_ref, task=task_id, meta=metadata)
        if isinstance(result.trace, CircuitTrace):
            return result.trace
        trace = result.trace
        features = _build_features(trace.get("features", []))
        edges = _build_edges(trace.get("edges", []))
        sparsity = float(trace.get("sparsity", 0.0))
        path_lengths = {k: float(v) for k, v in trace.get("path_lengths", {}).items()}
        return CircuitTrace(
            model_ref=str(trace.get("model_ref", model_ref)),
            task_id=str(trace.get("task_id", task_id)),
            features=features,
            edges=edges,
            sparsity=sparsity,
            path_lengths=path_lengths,
        )


__all__ = ["TraceSig", "TraceModule", "TraceRunner"]
