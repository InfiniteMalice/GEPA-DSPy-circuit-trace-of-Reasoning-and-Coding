"""Aggregation helpers for humanities rigor scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

try:  # pragma: no cover - optional during doc builds
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from .axes import HUMANITIES_AXES, HumanitiesScores, score_axis

DEFAULT_EPSILON = 1e-3


@dataclass(frozen=True)
class HumanitiesProfile:
    name: str
    weights: Mapping[str, float]

    def normalised_weights(self) -> Dict[str, float]:
        total = sum(self.weights.get(axis, 0.0) for axis in HUMANITIES_AXES)
        if total <= 0:
            raise ValueError(f"Profile {self.name} has no positive weights")
        return {
            axis: self.weights.get(axis, 0.0) / total for axis in HUMANITIES_AXES
        }


def _parse_profiles(text: str) -> Dict[str, HumanitiesProfile]:
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = {}
        current = None
        for raw in text.splitlines():
            if not raw.strip() or raw.strip().startswith("#"):
                continue
            if raw.startswith("profiles:"):
                data["profiles"] = {}
                continue
            if raw.startswith("  ") and raw.endswith(":"):
                current = raw.strip().rstrip(":")
                data.setdefault("profiles", {})[current] = {}
                continue
            if current and ":" in raw:
                axis, value = raw.split(":", 1)
                data["profiles"][current][axis.strip()] = float(value.strip())
    profiles: Dict[str, HumanitiesProfile] = {}
    for name, weights in data.get("profiles", {}).items():
        profiles[name] = HumanitiesProfile(name=name, weights=weights)
    return profiles


def load_profiles(path: str | Path | None = None) -> Dict[str, HumanitiesProfile]:
    if path is None:
        path = Path(__file__).with_name("profiles.yaml")
    return _parse_profiles(Path(path).read_text())


def score_axes(metrics: Mapping[str, Mapping[str, object]]) -> HumanitiesScores:
    scores = {
        axis: score_axis(axis, metrics.get(axis, {})) for axis in HUMANITIES_AXES
    }
    return HumanitiesScores(scores=scores)


def weighted_geometric_mean(
    scores: Mapping[str, int],
    weights: Mapping[str, float],
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    if not weights:
        raise ValueError("weights must not be empty")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("weights must sum to positive")
    log_sum = 0.0
    for axis, weight in weights.items():
        log_sum += (weight / total) * math.log(scores.get(axis, 0) + epsilon)
    return math.exp(log_sum)


def evaluate_profile(
    metrics: Mapping[str, Mapping[str, object]],
    profile: HumanitiesProfile,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> Dict[str, object]:
    scores = score_axes(metrics).scores
    weights = profile.normalised_weights()
    composite = weighted_geometric_mean(scores, weights, epsilon=epsilon)
    passes = HumanitiesScores(scores).passes_hard_gates()
    return {
        "profile": profile.name,
        "scores": scores,
        "composite": composite,
        "passes_gates": passes,
    }


__all__ = [
    "HumanitiesProfile",
    "load_profiles",
    "score_axes",
    "weighted_geometric_mean",
    "evaluate_profile",
]
