"""Profile-aware aggregation utilities for reasoning scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping

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


@dataclass(frozen=True)
class Profile:
    """Represents a named axis-weight profile."""

    name: str
    weights: Mapping[str, float]

    def normalised_weights(self) -> Dict[str, float]:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError(f"Profile {self.name} has non-positive total weight")
        return {axis: weight / total for axis, weight in self.weights.items()}


def _parse_profiles(text: str) -> Dict[str, Profile]:
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = _fallback_parse(text)
    profiles: Dict[str, Profile] = {}
    for name, weights in data.get("profiles", {}).items():
        profiles[name] = Profile(name=name, weights=weights)
    return profiles


def _fallback_parse(text: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    result: Dict[str, Dict[str, Dict[str, float]]] = {"profiles": {}}
    current_profile: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.strip()
        if indent == 0 and line.endswith(":"):
            if line[:-1] != "profiles":
                raise ValueError("Unexpected top-level key in profiles fallback parser")
            continue
        if indent == 2 and line.endswith(":"):
            current_profile = line[:-1]
            result["profiles"][current_profile] = {}
            continue
        if indent == 4 and ":" in line and current_profile is not None:
            axis, value = line.split(":", 1)
            result["profiles"][current_profile][axis.strip()] = float(value.strip())
            continue
        raise ValueError(f"Unable to parse line: {raw_line}")
    return result


def load_profiles(path: str | Path | None = None) -> Dict[str, Profile]:
    """Load profiles from YAML and return mapping of profile name to ``Profile``."""
    if path is None:
        path = Path(__file__).with_name("profiles.yaml")
    else:
        path = Path(path)
    return _parse_profiles(path.read_text())


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
    "apply_hard_gates",
    "weighted_geometric_mean",
    "evaluate_profile",
    "rank_candidates",
    "DEFAULT_GATES",
    "DEFAULT_EPSILON",
]
