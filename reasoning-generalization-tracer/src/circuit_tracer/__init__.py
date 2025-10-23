"""Minimal circuit_tracer stub for offline testing."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional


def trace(
    *,
    model_ref: str,
    task_id: str,
    hooks: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    del hooks, metadata
    return {
        "model_ref": model_ref,
        "task_id": task_id,
        "features": [
            {"id": "F0", "layer": 0, "importance": 0.3, "tags": ["baseline"]},
            {"id": "F1", "layer": 1, "importance": 0.6, "tags": ["parity"]},
        ],
        "edges": [
            {"src": "F0", "dst": "F1", "weight": 0.7},
        ],
        "sparsity": 0.8,
        "path_lengths": {"mean": 2.0, "max": 2},
    }


__all__ = ["trace"]
