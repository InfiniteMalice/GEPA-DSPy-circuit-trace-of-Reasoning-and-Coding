"""Public schema for optional reasoning graph metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass
class ReasoningNode:
    """A public reasoning graph node emitted by a model or adapter."""

    id: str
    label: str
    kind: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ReasoningNode":
        node_id = str(data.get("id") or "")
        if not node_id:
            raise ValueError("ReasoningNode requires non-empty 'id'")
        return cls(
            id=node_id,
            label=str(data.get("label", "")),
            kind=str(data.get("kind", "")),
            text=str(data.get("text", "")),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ReasoningEdge:
    """A typed directed relation between reasoning nodes."""

    source: str
    target: str
    relation: str
    weight: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "metadata": dict(self.metadata),
        }
        if self.weight is not None:
            payload["weight"] = float(self.weight)
        return payload

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ReasoningEdge":
        source = str(data.get("source") or "")
        target = str(data.get("target") or "")
        relation = str(data.get("relation") or "")
        if not source or not target:
            raise ValueError("ReasoningEdge requires non-empty endpoints")
        weight_value = data.get("weight")
        return cls(
            source=source,
            target=target,
            relation=relation,
            weight=float(weight_value) if weight_value is not None else None,
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class ReasoningGraph:
    """A graph of public reasoning metadata."""

    nodes: List[ReasoningNode]
    edges: List[ReasoningEdge]
    task_id: str | None = None
    answer_ref: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": dict(self.metadata),
        }
        if self.task_id is not None:
            payload["task_id"] = self.task_id
        if self.answer_ref is not None:
            payload["answer_ref"] = self.answer_ref
        return payload

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ReasoningGraph":
        return cls(
            nodes=[ReasoningNode.from_mapping(item) for item in data.get("nodes", [])],
            edges=[ReasoningEdge.from_mapping(item) for item in data.get("edges", [])],
            task_id=data.get("task_id"),
            answer_ref=data.get("answer_ref"),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "ReasoningGraph":
        return cls.from_mapping(json.loads(text))


__all__ = ["ReasoningEdge", "ReasoningGraph", "ReasoningNode"]
