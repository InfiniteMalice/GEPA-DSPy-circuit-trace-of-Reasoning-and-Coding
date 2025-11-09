"""Utility helpers shared across attribution modules."""

from __future__ import annotations

from typing import Any


def safe_int(value: Any, default: int = 0) -> int:
    """Return ``value`` coerced to ``int`` or ``default`` when conversion fails."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Return ``value`` coerced to ``float`` or ``default`` when conversion fails."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = ["safe_int", "safe_float"]
