"""Profile-aware aggregation utilities for reasoning scores."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover
    yaml = None

DEFAULT_EPSILON = 1e-3
DEFAULT_GATES = {
    "logical_validity": 3,
    "rigor": 3,
    "numerical_accuracy": 2,
}
_LAST_CONFIG: Dict[str, object] = {}


@dataclass(frozen=True)
class Profile:
    """Represents a named axis-weight profile."""

    name: str
    weights: Mapping[str, float]
    bonuses: Mapping[str, float] = field(default_factory=dict)

    def normalised_weights(self) -> Dict[str, float]:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError(f"Profile {self.name} has non-positive total weight")
        return {axis: weight / total for axis, weight in self.weights.items()}


def _parse_profiles(text: str) -> tuple[Dict[str, Profile], Dict[str, object]]:
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = _fallback_parse(text)
    if not isinstance(data, Mapping):
        raise ValueError("Profiles configuration must be a mapping")
    profiles: Dict[str, Profile] = {}
    raw_profiles = data.get("profiles") or {}
    if raw_profiles and not isinstance(raw_profiles, Mapping):
        raise ValueError("profiles section must be a mapping")
    for name, raw in raw_profiles.items():
        weights, bonuses = _split_profile_payload(raw)
        profiles[name] = Profile(name=name, weights=weights, bonuses=bonuses)
    raw_config = data.get("config") or {}
    if raw_config and not isinstance(raw_config, Mapping):
        raise ValueError("config section must be a mapping")
    config = dict(raw_config)
    return profiles, config


def _fallback_parse(text: str) -> Dict[str, object]:
    result: Dict[str, object] = {"profiles": {}}
    current_profile: str | None = None
    current_subsection: str | None = None
    current_nested: str | None = None
    section: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.strip()
        if indent == 0 and line.endswith(":"):
            section = line[:-1]
            if section not in {"profiles", "config"}:
                raise ValueError("Unexpected top-level key in profiles fallback parser")
            if section == "config":
                result.setdefault("config", {})
            continue
        # State: section header at indent 2 introduces a new profile entry.
        if section == "profiles" and indent == 2 and line.endswith(":"):
            current_profile = line[:-1]
            current_subsection = None
            result["profiles"][current_profile] = {}
            continue
        # State: indent 4 under profiles marks nested subsection (e.g. bonuses).
        if section == "profiles" and indent == 4 and line.endswith(":") and current_profile:
            current_subsection = line[:-1]
            result["profiles"][current_profile][current_subsection] = {}
            continue
        if section == "profiles" and indent >= 4 and ":" in line and current_profile:
            if indent == 4:
                current_subsection = None
            axis, value = line.split(":", 1)
            target = result["profiles"][current_profile]
            if current_subsection is not None and indent > 4:
                target = target[current_subsection]
            value = value.strip()
            if value:
                try:
                    target[axis.strip()] = float(value)
                except ValueError:
                    target[axis.strip()] = value
            else:
                target[axis.strip()] = value
            continue
        # State: config entries at indent 2 define new subsections or scalar keys.
        if section == "config" and indent == 2 and line.endswith(":"):
            current_profile = None
            current_subsection = line[:-1]
            current_nested = None
            result.setdefault("config", {})[current_subsection] = {}
            continue
        if section == "config" and indent == 2 and ":" in line and not line.endswith(":"):
            key, value = line.split(":", 1)
            result.setdefault("config", {})[key.strip()] = _parse_scalar_value(value)
            continue
        if section == "config" and indent == 4 and line.endswith(":"):
            container = result.setdefault("config", {})
            if current_subsection is not None:
                container = container.setdefault(current_subsection, {})
            current_nested = line[:-1]
            container[current_nested] = {}
            continue
        # State: indent 4 under config writes nested key/value pairs.
        if section == "config" and indent == 4 and ":" in line and not line.endswith(":"):
            current_nested = None
            key, value = line.split(":", 1)
            config_section = result.setdefault("config", {})
            if current_subsection is not None:
                config_section = config_section.setdefault(current_subsection, {})
            config_section[key.strip()] = _parse_scalar_value(value)
            continue
        if section == "config" and indent > 4 and ":" in line:
            key, value = line.split(":", 1)
            container = result.setdefault("config", {})
            if current_subsection is not None:
                container = container.setdefault(current_subsection, {})
            if current_nested is not None:
                container = container.setdefault(current_nested, {})
            container[key.strip()] = _parse_scalar_value(value)
            continue
        raise ValueError(f"Unable to parse line: {raw_line}")
    return result


def _split_profile_payload(
    raw: Mapping[str, object] | Iterable[tuple[str, object]],
) -> tuple[Dict[str, float], Dict[str, float]]:
    if isinstance(raw, Mapping):
        bonuses_raw = raw.get("bonuses", {})
        if "weights" in raw and isinstance(raw["weights"], Mapping):
            weights_source = raw["weights"].items()
        else:
            weights_source = (
                (key, value) for key, value in raw.items() if _is_number(value) and key != "bonuses"
            )
    else:
        bonuses_raw = {}
        weights_source = raw
    weights = {axis: float(value) for axis, value in weights_source}
    bonuses: Dict[str, float] = {}
    if isinstance(bonuses_raw, Mapping):
        for axis, value in bonuses_raw.items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip().lower() == "null":
                continue
            try:
                bonuses[str(axis)] = float(value)
            except (TypeError, ValueError):
                continue
    return weights, bonuses


def _is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        try:
            float(stripped)
        except ValueError:
            return False
        return True
    return False


def _parse_scalar_value(text: str) -> object:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] in {'"', "'"} and stripped[-1] == stripped[0]:
        return stripped[1:-1]
    value = stripped
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_profiles(path: str | Path | None = None) -> Dict[str, Profile]:
    """Load profiles from YAML and return mapping of profile name to ``Profile``."""
    if path is None:
        path = Path(__file__).with_name("profiles.yaml")
    else:
        path = Path(path)
    profiles, config = _parse_profiles(path.read_text())
    global _LAST_CONFIG
    _LAST_CONFIG = copy.deepcopy(config)
    return profiles


def get_last_config() -> Dict[str, object]:
    """Return the configuration parsed during :func:`load_profiles`."""

    return copy.deepcopy(_LAST_CONFIG)


def apply_hard_gates(
    axis_scores: Mapping[str, float],
    gates: Mapping[str, float] | None = None,
) -> tuple[bool, Dict[str, float]]:
    """Return whether all gates pass and the subset of gate scores."""
    if gates is None:
        gates = DEFAULT_GATES
    failed: Dict[str, float] = {}
    for axis, threshold in gates.items():
        score = axis_scores.get(axis, 0.0)
        if score < threshold:
            failed[axis] = score
    return (len(failed) == 0, failed)


def weighted_geometric_mean(
    axis_scores: Mapping[str, float],
    weights: Mapping[str, float],
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Compute the weighted geometric mean of ``axis_scores``."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not weights:
        raise ValueError("weights must not be empty")
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("total weight must be positive")
    log_sum = 0.0
    for axis, weight in weights.items():
        score = axis_scores.get(axis, 0.0)
        log_sum += (weight / total_weight) * math.log(score + epsilon)
    return math.exp(log_sum)


def evaluate_profile(
    axis_scores: Mapping[str, float],
    profile: Profile,
    gates: Mapping[str, float] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> Dict[str, object]:
    """Evaluate a profile returning composite score and gate diagnostics."""
    weights = profile.normalised_weights()
    composite = weighted_geometric_mean(axis_scores, weights, epsilon=epsilon)
    passes, failed = apply_hard_gates(axis_scores, gates)
    return {
        "profile": profile.name,
        "composite": composite,
        "passes_gates": passes,
        "failed_gates": failed,
    }


def rank_candidates(
    candidates: Iterable[Mapping[str, float]],
    profile: Profile,
    gates: Mapping[str, float] | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> list[Dict[str, object]]:
    """Evaluate and sort candidates by composite score descending."""
    results = [
        {**evaluate_profile(scores, profile, gates=gates, epsilon=epsilon), "scores": dict(scores)}
        for scores in candidates
    ]
    return sorted(results, key=lambda item: item["composite"], reverse=True)


__all__ = [
    "Profile",
    "load_profiles",
    "get_last_config",
    "apply_hard_gates",
    "weighted_geometric_mean",
    "evaluate_profile",
    "rank_candidates",
    "DEFAULT_GATES",
    "DEFAULT_EPSILON",
]
