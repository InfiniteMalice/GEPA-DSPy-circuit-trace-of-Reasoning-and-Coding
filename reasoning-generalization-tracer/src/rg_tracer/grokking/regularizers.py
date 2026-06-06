"""LRD-inspired spectral regularization for selected toy-transformer matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .config import SpectralRegularizationConfig
from .spectral_metrics import require_torch, singular_value_summary


def polar_factor_exact(weight: Any) -> Any:
    """Return the exact SVD polar factor U @ Vh for a small matrix."""

    torch_mod = require_torch()
    if weight.ndim != 2:
        raise ValueError("polar_factor_exact requires a 2D matrix")
    u, _, vh = torch_mod.linalg.svd(weight.detach().float(), full_matrices=False)
    return u @ vh


def polar_factor_newton_schulz(weight: Any, steps: int, eps: float = 1e-6) -> Any:
    """Approximate the polar factor with a bounded Newton-Schulz iteration."""

    torch_mod = require_torch()
    if weight.ndim != 2:
        raise ValueError("polar_factor_newton_schulz requires a 2D matrix")
    if steps <= 0:
        raise ValueError("steps must be positive")
    matrix = weight.detach().float()
    norm = torch_mod.linalg.matrix_norm(matrix, ord=2)
    if float(norm) <= eps:
        return torch_mod.zeros_like(matrix)
    x_value = matrix / torch_mod.clamp(norm, min=eps)
    eye = torch_mod.eye(x_value.shape[1], device=x_value.device, dtype=x_value.dtype)
    for _ in range(steps):
        xtx = x_value.transpose(-2, -1) @ x_value
        x_value = 0.5 * x_value @ (3.0 * eye - xtx)
    return torch_mod.nan_to_num(x_value)


def _target_selected(name: str, config: SpectralRegularizationConfig) -> bool:
    return name.endswith(".weight") and any(target in name for target in config.target_matrices)


def _scale_update(weight: Any, update: Any, config: SpectralRegularizationConfig) -> Any:
    if not config.adaptive_scale:
        return update
    torch_mod = require_torch()
    weight_norm = torch_mod.linalg.vector_norm(weight.detach())
    update_norm = torch_mod.linalg.vector_norm(update)
    if float(update_norm) <= config.eps:
        return update
    max_norm = torch_mod.clamp(weight_norm, min=config.eps)
    return update * torch_mod.clamp(max_norm / update_norm, max=1.0)


def _empty_diagnostic(name: str, mode: str) -> dict[str, object]:
    return {
        "matrix_name": name,
        "mode": mode,
        "frobenius_norm": 0.0,
        "nuclear_norm": 0.0,
        "spectral_norm": 0.0,
        "effective_rank": 0.0,
        "stable_rank": 0.0,
        "top_singular_values": [],
        "polar_approximation_error": None,
        "update_norm": 0.0,
    }


def _diagnostic(
    name: str, weight: Any, update: Any, polar_error: float | None
) -> dict[str, object]:
    torch_mod = require_torch()
    summary = singular_value_summary(weight)
    return {
        "matrix_name": name,
        "frobenius_norm": summary["frobenius_norm"],
        "nuclear_norm": summary["nuclear_norm"],
        "spectral_norm": summary["spectral_norm"],
        "effective_rank": summary["effective_rank"],
        "stable_rank": summary["stable_rank"],
        "top_singular_values": summary["top_singular_values"],
        "polar_approximation_error": polar_error,
        "update_norm": float(torch_mod.linalg.vector_norm(update).item()),
    }


def apply_low_rank_decay_(
    named_parameters: Iterable[tuple[str, Any]],
    config: SpectralRegularizationConfig,
) -> list[dict[str, object]]:
    """Apply decoupled L2/LRD updates in-place and return JSON-safe diagnostics."""

    torch_mod = require_torch()
    diagnostics: list[dict[str, object]] = []
    if config.mode == "none":
        return diagnostics
    with torch_mod.no_grad():
        for name, parameter in named_parameters:
            if parameter.ndim != 2 or not _target_selected(name, config):
                continue
            update = torch_mod.zeros_like(parameter)
            polar_error = None
            if config.mode in {"l2", "l2_plus_lrd"}:
                update = update + config.l2_weight * parameter.detach()
            if config.mode in {"lrd", "l2_plus_lrd"}:
                exact = polar_factor_exact(parameter)
                if config.use_newton_schulz:
                    polar = polar_factor_newton_schulz(
                        parameter,
                        steps=config.newton_schulz_steps,
                        eps=config.eps,
                    )
                    diff = torch_mod.linalg.vector_norm(exact - polar)
                    polar_error = float(diff.item())
                else:
                    polar = exact
                    polar_error = 0.0
                update = update + config.lrd_weight * _scale_update(parameter, polar, config)
            if float(torch_mod.linalg.vector_norm(update).item()) > 0.0:
                parameter.sub_(update.to(parameter.dtype))
            payload = _diagnostic(name, parameter.detach(), update.detach(), polar_error)
            payload["mode"] = config.mode
            diagnostics.append(payload)
    return diagnostics


@dataclass
class LowRankDecay:
    """Optional LRD-inspired spectral regularizer for selected toy-transformer matrices."""

    config: SpectralRegularizationConfig

    def step(self, named_parameters: Iterable[tuple[str, Any]]) -> list[dict[str, object]]:
        """Apply one decoupled regularization step."""

        return apply_low_rank_decay_(named_parameters, self.config)


__all__ = [
    "LowRankDecay",
    "apply_low_rank_decay_",
    "polar_factor_exact",
    "polar_factor_newton_schulz",
]
