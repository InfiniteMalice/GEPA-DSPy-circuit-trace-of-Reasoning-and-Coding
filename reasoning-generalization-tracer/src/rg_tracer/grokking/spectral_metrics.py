"""Real spectral metrics for toy-transformer matrices."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised when the optional extra is absent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def require_torch() -> Any:
    """Return Torch or raise a clear optional dependency error."""

    if torch is None:
        raise ImportError(
            "rg_tracer.grokking requires the optional extra: pip install -e .[grokking]"
        )
    return torch


def _singular_values(weight: Any) -> Any:
    torch_mod = require_torch()
    if weight.ndim != 2:
        raise ValueError("spectral metrics require a 2D matrix")
    values = torch_mod.linalg.svdvals(weight.detach().float())
    return torch_mod.clamp(values, min=0.0)


def spectral_entropy(weight: Any, eps: float = 1e-12) -> float:
    """Return entropy of normalized singular values."""

    torch_mod = require_torch()
    values = _singular_values(weight)
    total = values.sum()
    if float(total) <= eps:
        return 0.0
    probs = values / torch_mod.clamp(total, min=eps)
    entropy = -(probs * torch_mod.log(torch_mod.clamp(probs, min=eps))).sum()
    return float(entropy.item())


def effective_rank(weight: Any, eps: float = 1e-12) -> float:
    """Return exp(spectral entropy), with zero matrices mapped to 0."""

    values = _singular_values(weight)
    if float(values.sum()) <= eps:
        return 0.0
    torch_mod = require_torch()
    return float(torch_mod.exp(torch_mod.tensor(spectral_entropy(weight, eps))).item())


def stable_rank(weight: Any, eps: float = 1e-12) -> float:
    """Return squared Frobenius norm divided by squared spectral norm."""

    values = _singular_values(weight)
    if values.numel() == 0:
        return 0.0
    spectral = values.max()
    if float(spectral) <= eps:
        return 0.0
    return float(((values.pow(2).sum()) / (spectral.pow(2) + eps)).item())


def singular_value_summary(weight: Any, top_k: int = 8) -> dict[str, object]:
    """Return JSON-safe singular-value metrics from an actual matrix."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    values = _singular_values(weight)
    return {
        "effective_rank": effective_rank(weight),
        "stable_rank": stable_rank(weight),
        "spectral_entropy": spectral_entropy(weight),
        "spectral_norm": float(values.max().item()) if values.numel() else 0.0,
        "nuclear_norm": float(values.sum().item()),
        "frobenius_norm": float(values.pow(2).sum().sqrt().item()),
        "top_singular_values": [float(item) for item in values[:top_k].tolist()],
    }


def rank_collapse_ratio(initial: Any, current: Any, eps: float = 1e-12) -> float:
    """Return 1 - current effective rank / initial effective rank."""

    initial_rank = effective_rank(initial, eps)
    if initial_rank <= eps:
        return 0.0
    current_rank = effective_rank(current, eps)
    return float(max(0.0, min(1.0, 1.0 - current_rank / initial_rank)))


__all__ = [
    "effective_rank",
    "rank_collapse_ratio",
    "require_torch",
    "singular_value_summary",
    "spectral_entropy",
    "stable_rank",
]
