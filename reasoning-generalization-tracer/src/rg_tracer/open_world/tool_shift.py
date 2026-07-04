"""Tool/action schema perturbations."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def rename_tool(tool: Mapping[str, Any], new_name: str) -> Dict[str, Any]:
    shifted = dict(tool)
    old_name = str(shifted.get("name", "unknown"))
    shifted["name"] = new_name
    shifted["metadata"] = {
        **dict(shifted.get("metadata", {}) or {}),
        "shift_type": "tool_rename",
        "old_name": old_name,
    }
    return shifted


def reorder_schema_fields(schema: Mapping[str, Any]) -> Dict[str, Any]:
    shifted = dict(schema)
    properties = dict(shifted.get("properties", {}) or {})
    shifted["properties"] = {key: properties[key] for key in sorted(properties)}
    shifted["metadata"] = {
        **dict(shifted.get("metadata", {}) or {}),
        "shift_type": "schema_reorder",
        "schema_drift": True,
    }
    return shifted


def add_optional_field(
    schema: Mapping[str, Any], name: str, spec: Mapping[str, Any]
) -> Dict[str, Any]:
    shifted = dict(schema)
    properties = dict(shifted.get("properties", {}) or {})
    properties[name] = dict(spec)
    shifted["properties"] = properties
    shifted["metadata"] = {
        **dict(shifted.get("metadata", {}) or {}),
        "shift_type": "schema_add_optional",
        "schema_drift": True,
    }
    return shifted


def add_distractor_tool(tool: Mapping[str, Any], distractor_name: str) -> Dict[str, Any]:
    return {
        "tools": [dict(tool), {"name": distractor_name, "description": "Distractor tool"}],
        "metadata": {"shift_type": "tool_distractor"},
    }


__all__ = ["add_distractor_tool", "add_optional_field", "rename_tool", "reorder_schema_fields"]
