"""Reasoning and generalization tracing toolkit."""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - during local development
    __version__ = version("rg-tracer")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
