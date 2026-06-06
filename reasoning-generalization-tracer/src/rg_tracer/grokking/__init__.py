"""Optional toy grokking experiments for RG-Tracer.

The package is importable without Torch. Modules that need real tensor operations
raise a clear optional-extra error when Torch is unavailable.
"""

from __future__ import annotations

from .config import SpectralRegularizationConfig, ToyGrokkingConfig

__all__ = ["SpectralRegularizationConfig", "ToyGrokkingConfig"]
