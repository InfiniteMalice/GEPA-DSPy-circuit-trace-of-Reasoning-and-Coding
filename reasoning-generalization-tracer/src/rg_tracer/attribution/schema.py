"""Normalised schema helpers for attribution graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping


@dataclass
class GraphNode:
    """Represents a node within an attribution graph."""

    id: str
    layer: int
    type: str
    activation: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "layer": int(self.layer),
            "type": self.type,
            "activation": float(self.activation),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GraphNode":
        return cls(
            id=str(data.get("id", "")),
            layer=int(data.get("layer", 0)),
            type=str(data.get("type", "unknown")),
            activation=float(data.get("activation", 0.0)),
        )


@dataclass
class GraphEdge:
    """Represents a directed attribution edge between nodes."""

    src: str
    dst: str
    attr: float

    def to_dict(self) -> Dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "attr": float(self.attr)}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GraphEdge":
        return cls(
            src=str(data.get("src", "")),
            dst=str(data.get("dst", "")),
            attr=float(data.get("attr", 0.0)),
        )


@dataclass
class GraphMeta:
    """Metadata describing the attribution run."""

    token_positions: List[int] = field(default_factory=list)
    logits_scale: float | None = None
    phase: str | None = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"token_positions": list(self.token_positions)}
        if self.logits_scale is not None:
            payload["logits_scale"] = float(self.logits_scale)
        if self.phase is not None:
            payload["phase"] = str(self.phase)
        payload.update(self.extras)
        return payload

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GraphMeta":
        meta = cls()
        if "token_positions" in data:
            meta.token_positions = [int(value) for value in data.get("token_positions", [])]
        if data.get("logits_scale") is not None:
            meta.logits_scale = float(data.get("logits_scale"))
        if data.get("phase") is not None:
            meta.phase = str(data.get("phase"))
        extras = {
            key: value
            for key, value in data.items()
            if key not in {"token_positions", "logits_scale", "phase"}
        }
        meta.extras = dict(extras)
        return meta


@dataclass
class AttributionGraph:
    """Container holding nodes, edges, and metadata for a single probe."""

    model_ref: str
    task_id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    meta: GraphMeta = field(default_factory=GraphMeta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_ref": self.model_ref,
            "task_id": self.task_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "meta": self.meta.to_dict(),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AttributionGraph":
        nodes = [GraphNode.from_mapping(item) for item in data.get("nodes", [])]
        edges = [GraphEdge.from_mapping(item) for item in data.get("edges", [])]
        meta = GraphMeta.from_mapping(data.get("meta", {}))
        return cls(
            model_ref=str(data.get("model_ref", "unknown")),
            task_id=str(data.get("task_id", "unknown")),
            nodes=nodes,
            edges=edges,
            meta=meta,
        )


def normalise_graph(data: Mapping[str, Any]) -> AttributionGraph:
    """Return a normalised :class:`AttributionGraph` from arbitrary mapping input."""

    return AttributionGraph.from_mapping(data)


def merge_graphs(graphs: Iterable[Mapping[str, Any]]) -> AttributionGraph:
    """Merge multiple graphs by averaging edge attributions and activations."""

    graphs = list(graphs)
    if not graphs:
        return AttributionGraph(model_ref="unknown", task_id="unknown", nodes=[], edges=[])
    normalised = [normalise_graph(graph) for graph in graphs]
    base = normalised[0]
    node_attrs: MutableMapping[str, List[float]] = {}
    edge_attrs: MutableMapping[tuple[str, str], List[float]] = {}
    for graph in normalised:
        for node in graph.nodes:
            node_attrs.setdefault(node.id, []).append(node.activation)
        for edge in graph.edges:
            key = (edge.src, edge.dst)
            edge_attrs.setdefault(key, []).append(edge.attr)
    node_lookup = {node.id: node for node in base.nodes}
    merged_nodes = []
    for node_id, values in node_attrs.items():
        template = node_lookup.get(node_id)
        layer = template.layer if template is not None else 0
        node_type = template.type if template is not None else "unknown"
        merged_nodes.append(
            GraphNode(
                id=node_id,
                layer=layer,
                type=node_type,
                activation=sum(values) / len(values),
            )
        )
    merged_edges = [
        GraphEdge(src=src, dst=dst, attr=sum(values) / len(values))
        for (src, dst), values in edge_attrs.items()
    ]
    meta = base.meta
    meta.extras.setdefault("merged_count", len(normalised))
    return AttributionGraph(
        model_ref=base.model_ref,
        task_id=base.task_id,
        nodes=merged_nodes,
        edges=merged_edges,
        meta=meta,
    )


__all__ = [
    "GraphNode",
    "GraphEdge",
    "GraphMeta",
    "AttributionGraph",
    "normalise_graph",
    "merge_graphs",
]
