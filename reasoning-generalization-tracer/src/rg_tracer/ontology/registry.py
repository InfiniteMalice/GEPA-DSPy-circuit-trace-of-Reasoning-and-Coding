"""Load the packaged ontology vocabulary, independently of the behavioral contract."""

from __future__ import annotations

import json
from copy import deepcopy
from enum import Enum
from importlib.resources import files
from typing import Any


def load_registry() -> dict[str, Any]:
    """Return a fresh registry with expanded, explicit relation semantics."""
    data = json.loads(files(__package__).joinpath("registry.json").read_text(encoding="utf-8"))
    types = set(data["entity_types"])
    for name, definition in data["relations"].items():
        for key, value in data["relation_defaults"].items():
            definition.setdefault(key, value)
        signatures = []
        for view, source, target in definition["signatures"]:
            domain = data["groups"].get(source, [source])
            range_types = data["groups"].get(target, [target])
            if not set(domain + range_types) <= types or view not in data["views"]:
                raise ValueError(f"Invalid registry signature for {name}")
            signatures.append(
                {
                    "view": view,
                    "domain": domain,
                    "range": range_types,
                    "cycles_permitted": not data["views"][view]["acyclic"],
                }
            )
        definition["signatures"] = signatures
    return data


_REGISTRY = load_registry()
ONTOLOGY_VERSION = _REGISTRY["ontology_version"]
EntityType = Enum("EntityType", {key: key for key in _REGISTRY["entity_types"]}, type=str)
RelationType = Enum("RelationType", {key: key for key in _REGISTRY["relations"]}, type=str)
EpistemicStatus = Enum(
    "EpistemicStatus", {key: key for key in _REGISTRY["epistemic_statuses"]}, type=str
)
GraphView = Enum("GraphView", {key.upper(): key for key in _REGISTRY["views"]}, type=str)


def relation_definition(relation: str) -> dict[str, Any]:
    """Resolve documented spelling aliases without defining extra relation meanings."""
    key = _REGISTRY["relation_aliases"].get(relation, relation)
    try:
        return deepcopy(_REGISTRY["relations"][key])
    except KeyError as error:
        raise ValueError(f"Unregistered ontology relation: {relation}") from error
