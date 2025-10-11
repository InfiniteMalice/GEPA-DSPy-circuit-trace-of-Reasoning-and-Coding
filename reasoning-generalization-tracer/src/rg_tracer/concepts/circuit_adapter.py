"""Circuit tracer adapter producing a normalised JSON schema."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:  # pragma: no cover - dependency optional in tests
    from circuit_tracer import trace as ct_trace
except Exception:  # pragma: no cover
    ct_trace = None


@dataclass
class CircuitFeature:
    id: str
    layer: int
    importance: float
    tags: List[str]


@dataclass
class CircuitEdge:
    src: str
    dst: str
    weight: float


@dataclass
class CircuitTrace:
    model_ref: str
    task_id: str
    features: List[CircuitFeature]
    edges: List[CircuitEdge]
    sparsity: float
    path_lengths: Dict[str, float]

    def to_json(self) -> Dict[str, Any]:
        return {
            "model_ref": self.model_ref,
            "task_id": self.task_id,
            "features": [asdict(feature) for feature in self.features],
            "edges": [asdict(edge) for edge in self.edges],
            "sparsity": self.sparsity,
            "path_lengths": dict(self.path_lengths),
        }


def _normalise_trace(raw: Mapping[str, Any]) -> CircuitTrace:
    features = [
        CircuitFeature(
            id=str(feat.get("id")),
            layer=int(feat.get("layer", 0)),
            importance=float(feat.get("importance", 0.0)),
            tags=list(feat.get("tags", [])),
        )
        for feat in raw.get("features", [])
    ]
    edges = [
        CircuitEdge(
            src=str(edge.get("src")),
            dst=str(edge.get("dst")),
            weight=float(edge.get("weight", 0.0)),
        )
        for edge in raw.get("edges", [])
    ]
    sparsity = float(raw.get("sparsity", 0.0))
    path_lengths = {
        key: float(value)
        for key, value in raw.get("path_lengths", {}).items()
    }
    return CircuitTrace(
        model_ref=str(raw.get("model_ref", "unknown")),
        task_id=str(raw.get("task_id", "unknown")),
        features=features,
        edges=edges,
        sparsity=sparsity,
        path_lengths=path_lengths,
    )


def trace_model(
    model_ref: str,
    task_id: str,
    *,
    hooks: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> CircuitTrace:
    """Invoke circuit-tracer if available, otherwise return a stub trace."""
    if ct_trace is None:
        # produce a deterministic stub trace for testing
        features = [
            CircuitFeature(id="F0", layer=0, importance=0.2, tags=["baseline"]),
            CircuitFeature(id="F1", layer=1, importance=0.5, tags=["parity"]),
        ]
        edges = [CircuitEdge(src="F0", dst="F1", weight=0.8)]
        path_lengths = {"mean": 2.0, "max": 2.0}
        return CircuitTrace(
            model_ref=model_ref,
            task_id=task_id,
            features=features,
            edges=edges,
            sparsity=0.8,
            path_lengths=path_lengths,
        )
    raw = ct_trace(model_ref=model_ref, task_id=task_id, hooks=hooks, metadata=metadata)
    return _normalise_trace(raw)


__all__ = ["CircuitTrace", "CircuitFeature", "CircuitEdge", "trace_model"]
