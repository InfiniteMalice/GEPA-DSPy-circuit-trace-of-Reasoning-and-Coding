"""Observation-shift perturbations."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def add_redundant_observation(observation: Mapping[str, Any], key: str) -> Dict[str, Any]:
    shifted = dict(observation)
    shifted[f"redundant_{key}"] = observation.get(key)
    shifted["metadata"] = {
        **dict(shifted.get("metadata", {}) or {}),
        "shift_type": "observation_redundant",
    }
    return shifted


def add_stale_observation(observation: Mapping[str, Any], stale_value: Any) -> Dict[str, Any]:
    shifted = dict(observation)
    shifted["stale_observation"] = stale_value
    shifted["metadata"] = {
        **dict(shifted.get("metadata", {}) or {}),
        "shift_type": "observation_stale",
        "requires_verification": True,
    }
    return shifted


def change_value_format(observation: Mapping[str, Any], key: str) -> Dict[str, Any]:
    shifted = dict(observation)
    if key in shifted:
        shifted[key] = str(shifted[key])
    shifted["metadata"] = {
        **dict(shifted.get("metadata", {}) or {}),
        "shift_type": "observation_format_change",
    }
    return shifted


def inject_anomaly(observation: Mapping[str, Any], anomaly: str) -> Dict[str, Any]:
    shifted = dict(observation)
    shifted["anomaly"] = anomaly
    shifted["metadata"] = {
        **dict(shifted.get("metadata", {}) or {}),
        "shift_type": "observation_anomaly",
        "requires_verification": True,
    }
    return shifted


__all__ = [
    "add_redundant_observation",
    "add_stale_observation",
    "change_value_format",
    "inject_anomaly",
]
