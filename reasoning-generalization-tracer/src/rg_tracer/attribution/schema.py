"""Normalised schema helpers for attribution graphs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping

from .utils import safe_float, safe_int


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
        node_id = str(data.get("id") or "")
        if not node_id:
            raise ValueError("GraphNode requires non-empty 'id'")
        return cls(
            id=node_id,
            layer=safe_int(data.get("layer"), 0),
            type=str(data.get("type", "unknown")),
            activation=safe_float(data.get("activation"), 0.0),
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
        src = str(data.get("src") or "")
        dst = str(data.get("dst") or "")
        if not src or not dst:
            raise ValueError("GraphEdge requires non-empty 'src' and 'dst'")
        return cls(
            src=src,
            dst=dst,
            attr=safe_float(data.get("attr"), 0.0),
        )


@dataclass
class GraphMeta:
    """Metadata describing the attribution run."""

    token_positions: List[int] = field(default_factory=list)
    logits_scale: float | None = None
    phase: str | None = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        reserved = {"token_positions", "logits_scale", "phase"}
        payload: Dict[str, Any] = {
            key: value for key, value in self.extras.items() if key not in reserved
        }
        payload["token_positions"] = list(self.token_positions)
        if self.logits_scale is not None:
            payload["logits_scale"] = float(self.logits_scale)
        if self.phase is not None:
            payload["phase"] = str(self.phase)
        return payload

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GraphMeta":
        meta = cls()
        positions_value = data.get("token_positions")
        if positions_value is not None:
            if isinstance(positions_value, Iterable) and not isinstance(
                positions_value, (str, bytes)
            ):
                parsed_positions: List[int] = []
                for value in positions_value:
                    try:
                        parsed_positions.append(int(value))
                    except (TypeError, ValueError):
                        continue
                meta.token_positions = parsed_positions
        logits_value = data.get("logits_scale")
        if logits_value is not None:
            try:
                meta.logits_scale = float(logits_value)
            except (TypeError, ValueError):
                meta.logits_scale = None
        phase_value = data.get("phase")
        if phase_value is not None:
            meta.phase = str(phase_value)
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

    graphs_list = list(graphs)
    if not graphs_list:
        return AttributionGraph(
            model_ref="unknown", task_id="unknown", nodes=[], edges=[]
        )
    normalised = [normalise_graph(graph) for graph in graphs_list]
    base = normalised[0]
    node_attrs: MutableMapping[str, List[float]] = {}
    edge_attrs: MutableMapping[tuple[str, str], List[float]] = {}
    node_templates: Dict[str, List[GraphNode]] = {}
    for graph in normalised:
        for node in graph.nodes:
            node_attrs.setdefault(node.id, []).append(node.activation)
            node_templates.setdefault(node.id, []).append(node)
        for edge in graph.edges:
            key = (edge.src, edge.dst)
            edge_attrs.setdefault(key, []).append(edge.attr)
    merged_nodes = []
    for node_id in sorted(node_attrs):
        values = node_attrs[node_id]
        templates = node_templates.get(node_id, [])
        if templates:
            layer_counts = Counter(template.layer for template in templates)
            layer = layer_counts.most_common(1)[0][0]
            type_candidates = [template.type for template in templates if template.type]
            if type_candidates:
                type_counts = Counter(type_candidates)
                node_type = type_counts.most_common(1)[0][0]
            else:
                node_type = "unknown"
        else:
            layer = 0
            node_type = "unknown"
        merged_nodes.append(
            GraphNode(
                id=node_id,
                layer=layer,
                type=node_type,
                activation=sum(values) / len(values),
            )
        )
    merged_edges = []
    for src, dst in sorted(edge_attrs):
        values = edge_attrs[(src, dst)]
        merged_edges.append(GraphEdge(src=src, dst=dst, attr=sum(values) / len(values)))
    meta = GraphMeta(
        token_positions=list(base.meta.token_positions),
        logits_scale=base.meta.logits_scale,
        phase=base.meta.phase,
        extras=dict(base.meta.extras),
    )
    meta.extras["merged_count"] = len(normalised)
    return AttributionGraph(
        model_ref=base.model_ref,
        task_id=base.task_id,
        nodes=merged_nodes,
        edges=merged_edges,
        meta=meta,
    )


__all__ = [
    "AttributionGraph",
    "GraphEdge",
    "GraphMeta",
    "GraphNode",
    "merge_graphs",
    "normalise_graph",
]
