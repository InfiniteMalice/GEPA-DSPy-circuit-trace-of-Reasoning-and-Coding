"""Pytest configuration to expose the package src directory."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
src_path = str(src_dir)
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def pytest_configure() -> None:
    """Ensure the ``src`` directory is importable for tests."""
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


@pytest.fixture(autouse=True)
def reset_aggregator_defaults():
    from rg_tracer.scoring import aggregator

    with aggregator._LAST_CONFIG_LOCK:
        aggregator._LAST_CONFIG = copy.deepcopy(aggregator.DEFAULT_CONFIG)
    yield
    with aggregator._LAST_CONFIG_LOCK:
        aggregator._LAST_CONFIG = copy.deepcopy(aggregator.DEFAULT_CONFIG)
