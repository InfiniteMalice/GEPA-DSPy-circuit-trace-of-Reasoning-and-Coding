"""PTRM-inspired bounded Gaussian test-time trajectory perturbations."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .types import PerturbationRecord, RefinementState


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _clamp_probability(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


@dataclass
class PerturbationConfig:
    """Configuration for bounded Gaussian inference-time exploration."""

    enabled: bool = False
    strategy: str = "gaussian"
    noise_std: float = 0.10
    min_noise: float = -0.25
    max_noise: float = 0.25
    perturb_after_depth: int = 1
    trajectories: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if self.strategy != "gaussian":
            raise ValueError("strategy must be 'gaussian'")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if self.min_noise > self.max_noise:
            raise ValueError("min_noise must be <= max_noise")
        for name in ("perturb_after_depth", "trajectories", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.perturb_after_depth < 0:
            raise ValueError("perturb_after_depth must be non-negative")
        if self.trajectories <= 0:
            raise ValueError("trajectories must be positive")


class GaussianTrajectoryPerturber:
    """Seeded bounded Gaussian perturbations for public toy refinement state."""

    def __init__(self, config: PerturbationConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)  # noqa: S311

    def maybe_perturb(
        self,
        state: RefinementState,
        *,
        trajectory_id: str,
        parent_id: str | None,
        depth: int,
    ) -> tuple[float, PerturbationRecord | None]:
        if not self.config.enabled or depth < self.config.perturb_after_depth:
            return 0.0, None
        sampled = self.rng.gauss(0.0, self.config.noise_std)
        bounded = _clamp(sampled, self.config.min_noise, self.config.max_noise)
        confidence_after = _clamp_probability(state.confidence + bounded * 0.25)
        uncertainty_after = _clamp_probability(state.uncertainty - bounded * 0.20)
        state_value_after = confidence_after - uncertainty_after
        record = PerturbationRecord(
            trajectory_id=trajectory_id,
            parent_id=parent_id,
            depth=depth,
            sampled_noise=sampled,
            bounded_noise=bounded,
            state_value_before=state.state_value,
            state_value_after=state_value_after,
            confidence_before=state.confidence,
            confidence_after=confidence_after,
            uncertainty_before=state.uncertainty,
            uncertainty_after=uncertainty_after,
        )
        return bounded, record


__all__ = ["GaussianTrajectoryPerturber", "PerturbationConfig"]
