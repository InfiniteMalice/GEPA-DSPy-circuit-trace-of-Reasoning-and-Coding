"""Configuration for experimental recursive refinement."""

from __future__ import annotations

from dataclasses import dataclass, field

MAX_DEPTH_LIMIT = 64
MAX_WIDTH_LIMIT = 32
MAX_TOTAL_UPDATES_LIMIT = 1024
VALID_ROUTE_STRATEGIES = {"adaptive", "round_robin", "fixed"}


def _validate_positive_int(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def _validate_probabilityish(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


@dataclass
class RefinementBudget:
    """Bounded CPU-friendly search budget for recursive refinement."""

    max_depth: int = 8
    max_width: int = 4
    min_depth: int = 1
    initial_width: int = 1
    branch_factor: int = 2
    prune_below: float = 0.0
    convergence_threshold: float = 0.95
    uncertainty_threshold: float = 0.25
    disagreement_threshold: float = 0.20
    max_total_updates: int = 64
    seed: int = 0

    def __post_init__(self) -> None:
        _validate_positive_int("max_depth", self.max_depth, MAX_DEPTH_LIMIT)
        _validate_positive_int("max_width", self.max_width, MAX_WIDTH_LIMIT)
        _validate_positive_int("min_depth", self.min_depth, self.max_depth)
        _validate_positive_int("initial_width", self.initial_width, self.max_width)
        _validate_positive_int("branch_factor", self.branch_factor, MAX_WIDTH_LIMIT)
        _validate_positive_int("max_total_updates", self.max_total_updates, MAX_TOTAL_UPDATES_LIMIT)
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth must be <= max_depth")
        if self.initial_width > self.max_width:
            raise ValueError("initial_width must be <= max_width")
        for name in (
            "prune_below",
            "convergence_threshold",
            "uncertainty_threshold",
            "disagreement_threshold",
        ):
            _validate_probabilityish(name, float(getattr(self, name)))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


@dataclass
class ViewRoutingConfig:
    """Controls MDT-inspired routing through public view operators."""

    enabled: bool = True
    max_active_views: int = 3
    revisit_views: bool = True
    route_strategy: str = "adaptive"

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a boolean")
        if type(self.revisit_views) is not bool:
            raise TypeError("revisit_views must be a boolean")
        _validate_positive_int("max_active_views", self.max_active_views, MAX_WIDTH_LIMIT)
        if self.route_strategy not in VALID_ROUTE_STRATEGIES:
            raise ValueError(f"route_strategy must be one of {sorted(VALID_ROUTE_STRATEGIES)}")


@dataclass
class RecursiveRefinementConfig:
    """Top-level configuration for the experimental GRAM/MDT-inspired scaffold."""

    budget: RefinementBudget = field(default_factory=RefinementBudget)
    routing: ViewRoutingConfig = field(default_factory=ViewRoutingConfig)
    adaptive_halting: bool = True
    progressive_widening: bool = True
    shared_prefix: bool = True
    process_reward_weight: float = 0.0
    log_trajectory_states: bool = True

    def __post_init__(self) -> None:
        for name in (
            "adaptive_halting",
            "progressive_widening",
            "shared_prefix",
            "log_trajectory_states",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        _validate_probabilityish("process_reward_weight", float(self.process_reward_weight))
