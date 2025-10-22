"""Pytest configuration to expose the package src directory."""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Ensure the ``src`` directory is importable for tests."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_path = str(src_dir)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
