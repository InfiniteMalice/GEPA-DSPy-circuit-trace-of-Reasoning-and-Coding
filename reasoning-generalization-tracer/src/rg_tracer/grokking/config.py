"""Configuration for optional toy grokking experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

VALID_REGULARIZATION_MODES = frozenset({"none", "l2", "lrd", "l2_plus_lrd"})


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nonnegative_float(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be a non-negative float")


@dataclass(frozen=True)
class ToyGrokkingConfig:
    """CPU-friendly modular-addition toy transformer settings."""

    modulus: int = 17
    train_fraction: float = 0.40
    epochs: int = 2000
    eval_interval: int = 25
    batch_size: int = 128
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 1
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        _validate_positive_int("modulus", self.modulus)
        if self.modulus < 3:
            raise ValueError("modulus must be at least 3")
        if not isinstance(self.train_fraction, (int, float)) or not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be strictly between 0 and 1")
        for name in ("epochs", "eval_interval", "batch_size", "d_model", "n_heads", "n_layers"):
            _validate_positive_int(name, getattr(self, name))
        _validate_nonnegative_float("learning_rate", self.learning_rate)
        if self.learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of cpu, cuda, or mps")


@dataclass(frozen=True)
class SpectralRegularizationConfig:
    """Settings for optional LRD-inspired spectral regularization."""

    mode: str = "none"
    l2_weight: float = 0.0
    lrd_weight: float = 0.0
    target_matrices: tuple[str, ...] = ("q_proj", "k_proj")
    newton_schulz_steps: int = 5
    eps: float = 1e-6
    adaptive_scale: bool = True
    use_newton_schulz: bool = False

    def __post_init__(self) -> None:
        if self.mode not in VALID_REGULARIZATION_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_REGULARIZATION_MODES)}")
        _validate_nonnegative_float("l2_weight", self.l2_weight)
        _validate_nonnegative_float("lrd_weight", self.lrd_weight)
        _validate_positive_int("newton_schulz_steps", self.newton_schulz_steps)
        _validate_nonnegative_float("eps", self.eps)
        if self.eps == 0.0:
            raise ValueError("eps must be positive")
        if isinstance(self.target_matrices, (str, bytes)) or not self.target_matrices:
            raise ValueError("target_matrices must not be empty")
        if any(not isinstance(name, str) or not name for name in self.target_matrices):
            raise ValueError("target_matrices must contain non-empty strings")
        if self.mode in {"lrd", "l2_plus_lrd"} and self.lrd_weight <= 0.0:
            raise ValueError("lrd modes require lrd_weight > 0")
        if self.mode in {"l2", "l2_plus_lrd"} and self.l2_weight <= 0.0:
            raise ValueError("l2 modes require l2_weight > 0")


__all__ = [
    "VALID_REGULARIZATION_MODES",
    "SpectralRegularizationConfig",
    "ToyGrokkingConfig",
]
