"""Global Response Normalization utilities."""

from __future__ import annotations

from collections.abc import Iterable

from .torch_stub import torch


def _to_list(data: object) -> list[object]:
    if hasattr(data, "tolist"):
        converted = data.tolist()  # type: ignore[union-attr]
        return converted if isinstance(converted, list) else [converted]
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        return list(data)
    return [data]


def _normalise_vector(values: list[float], eps: float) -> list[float]:
    """Normalize a vector by its RMS value: x / sqrt(mean(x^2) + eps)."""
    if not values:
        return values
    mean_sq = sum(value * value for value in values) / len(values)
    denom = (mean_sq + eps) ** 0.5
    return [value / denom for value in values]


def apply_grn(x: object, eps: float = 1e-6) -> "torch.Tensor":
    """Apply Global Response Normalization across the last dimension.

    Inputs can be a single vector or a batch of vectors.

    For single-element vectors, RMS normalization returns ``value / sqrt(value**2 + eps)``,
    which approaches the sign of the input for large magnitudes and approximately 1.0 when
    |value| ≈ 1. Callers relying on absolute scale should account for this behavior.
    """

    values = _to_list(x)
    if (
        values
        and isinstance(values[0], Iterable)
        and not isinstance(values[0], (str, bytes))
    ):
        normalised = [
            _normalise_vector([float(v) for v in row], eps)
            for row in values  # type: ignore[arg-type]
        ]
    else:
        normalised = _normalise_vector([float(v) for v in values], eps)
    return torch.tensor(normalised, dtype=getattr(torch, "float32", float))


__all__ = ["apply_grn"]
