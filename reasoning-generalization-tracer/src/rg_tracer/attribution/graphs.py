"""Adapters producing attribution graphs from models."""

from __future__ import annotations

import random
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - torch optional for CI
    import torch
except Exception:  # pragma: no cover
    torch = None

from .schema import AttributionGraph, GraphEdge, GraphMeta, GraphNode, merge_graphs


class AttributionBackend:
    """Base interface for attribution graph extraction backends."""

    def extract_graph(
        self,
        model: object,
        inputs: Iterable[Mapping[str, object]] | Mapping[str, object],
        *,
        layers: Sequence[int] | None = None,
        seed: int | None = None,
    ) -> Mapping[str, object]:
        raise NotImplementedError


class BackendNull(AttributionBackend):
    """Deterministic mock backend used for CI and documentation examples."""

    def __init__(self, *, noise: float = 0.01) -> None:
        self.noise = float(noise)

    def extract_graph(
        self,
        model: object,
        inputs: Iterable[Mapping[str, object]] | Mapping[str, object],
        *,
        layers: Sequence[int] | None = None,
        seed: int | None = None,
    ) -> Mapping[str, object]:
        rng = random.Random(seed or 0)
        samples = _coerce_sequence(inputs)
        graphs: list[AttributionGraph] = []
        model_type = getattr(model, "__class__", type("obj", (), {}))
        model_ref = getattr(model, "name", getattr(model_type, "__name__", "model"))
        for index, sample in enumerate(samples):
            task_id = str(sample.get("task_id", f"probe_{index}"))
            phase = str(sample.get("phase", "post_grok"))
            base_activation = 0.5 + 0.05 * index
            nodes = [
                GraphNode(
                    id="n_input",
                    layer=0,
                    type="token",
                    activation=1.0,
                ),
                GraphNode(
                    id="n_hidden",
                    layer=1,
                    type="mlp",
                    activation=base_activation,
                ),
                GraphNode(
                    id="n_output",
                    layer=2,
                    type="logit",
                    activation=base_activation + 0.1,
                ),
            ]
            if layers is not None:
                nodes = [node for node in nodes if node.layer in layers]
            node_ids = {node.id for node in nodes}
            edges = [
                GraphEdge(
                    src="n_input",
                    dst="n_hidden",
                    attr=0.6 + rng.random() * self.noise,
                ),
                GraphEdge(
                    src="n_hidden",
                    dst="n_output",
                    attr=0.3 + rng.random() * self.noise,
                ),
            ]
            if layers is not None:
                edges = [edge for edge in edges if edge.src in node_ids and edge.dst in node_ids]
            meta = GraphMeta(
                token_positions=list(range(len(sample.get("tokens", [0])))),
                logits_scale=1.0 + 0.1 * index,
                phase=phase,
                extras={"sample_index": index},
            )
            graphs.append(
                AttributionGraph(
                    model_ref=model_ref,
                    task_id=task_id,
                    nodes=nodes,
                    edges=edges,
                    meta=meta,
                )
            )
        if len(graphs) == 1:
            return graphs[0].to_dict()
        serialised: list[Mapping[str, object]] = []
        for graph in graphs:
            graph_dict: dict[str, object] = dict(graph.to_dict())
            meta_dict = graph_dict.get("meta")
            if isinstance(meta_dict, Mapping):
                meta_copy: dict[str, object] = dict(meta_dict)
                meta_copy.pop("sample_index", None)
                extras = meta_copy.get("extras")
                if isinstance(extras, Mapping):
                    extras_copy: dict[str, object] = dict(extras)
                    extras_copy.pop("sample_index", None)
                    meta_copy["extras"] = extras_copy
                graph_dict["meta"] = meta_copy
            serialised.append(graph_dict)
        merged = merge_graphs(serialised)
        return merged.to_dict()


class BackendHookedTransformer(AttributionBackend):
    "PyTorch backend using gradient x activation hooks."

    def __init__(self, *, reduce: str = "mean") -> None:
        self.reduce = reduce
        if torch is None:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is required for BackendHookedTransformer")

    def extract_graph(
        self,
        model: object,
        inputs: Iterable[Mapping[str, object]] | Mapping[str, object],
        *,
        layers: Sequence[int] | None = None,
        seed: int | None = None,
    ) -> Mapping[str, object]:
        raise NotImplementedError("Hooked transformer attribution is not yet implemented.")


class BackendExternal(AttributionBackend):
    "Placeholder backend for external attribution tooling."

    def __init__(self, *, endpoint: str | None = None) -> None:
        self.endpoint = endpoint or ""

    def extract_graph(
        self,
        model: object,
        inputs: Iterable[Mapping[str, object]] | Mapping[str, object],
        *,
        layers: Sequence[int] | None = None,
        seed: int | None = None,
    ) -> Mapping[str, object]:
        raise NotImplementedError(
            "External attribution integration must be provided by the caller."
        )


_BACKENDS: MutableMapping[str, Callable[..., AttributionBackend]] = {
    "null": BackendNull,
    "hooked": BackendHookedTransformer,
    "external": BackendExternal,
}


def register_backend(name: str, factory: Callable[..., AttributionBackend]) -> None:
    "Register a backend factory under ``name``."

    _BACKENDS[name] = factory


def get_backend(name: str, **kwargs: object) -> AttributionBackend:
    """Return an instantiated backend for ``name``."""

    if name not in _BACKENDS:
        raise KeyError(f"Unknown attribution backend: {name}")
    return _BACKENDS[name](**kwargs)


def extract_graph(
    model: object,
    inputs: Iterable[Mapping[str, object]] | Mapping[str, object],
    *,
    backend: AttributionBackend | None = None,
    backend_name: str = "null",
    layers: Sequence[int] | None = None,
    seed: int | None = None,
) -> Mapping[str, object]:
    "Extract an attribution graph using the configured backend."

    backend = backend or get_backend(backend_name)
    return backend.extract_graph(model, inputs, layers=layers, seed=seed)


def _coerce_sequence(
    inputs: Iterable[Mapping[str, object]] | Mapping[str, object],
) -> list[Mapping[str, object]]:
    if isinstance(inputs, Mapping):
        return [inputs]
    return list(inputs)


__all__ = [
    "AttributionBackend",
    "BackendExternal",
    "BackendHookedTransformer",
    "BackendNull",
    "extract_graph",
    "get_backend",
    "register_backend",
]
