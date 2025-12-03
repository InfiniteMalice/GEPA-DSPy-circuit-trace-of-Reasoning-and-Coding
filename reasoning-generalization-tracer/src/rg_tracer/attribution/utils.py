"""Utility helpers shared across attribution modules."""

from __future__ import annotations

import math
from typing import Any


def safe_int(value: Any, default: int = 0) -> int:
    """Return ``value`` coerced to ``int`` or ``default`` when conversion fails."""

    try:
        return int(value)
    except (OverflowError, TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Return ``value`` coerced to ``float`` or ``default`` when conversion fails."""

    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return default
    else:
        return result if math.isfinite(result) else default


__all__ = ["safe_float", "safe_int"]
