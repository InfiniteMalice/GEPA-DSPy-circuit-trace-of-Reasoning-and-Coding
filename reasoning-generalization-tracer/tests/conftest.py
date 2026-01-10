"""Pytest configuration to expose the package src directory."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from tests.gepa_test_stubs import install_gepa_stubs

repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
src_path = str(src_dir)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

install_gepa_stubs()


def _install_gepa_stubs() -> None:
    from rg_tracer.testing.gepa_stubs import install_gepa_stubs

    install_gepa_stubs()


_install_gepa_stubs()


def _install_gepa_stubs() -> None:
    from rg_tracer.testing.gepa_stubs import install_gepa_stubs

    install_gepa_stubs()


_install_gepa_stubs()


def _install_gepa_stubs() -> None:
    from rg_tracer.testing.gepa_stubs import install_gepa_stubs

    install_gepa_stubs()


_install_gepa_stubs()


def _install_gepa_stubs() -> None:
    from rg_tracer.testing.gepa_stubs import install_gepa_stubs

    install_gepa_stubs()


_install_gepa_stubs()


def _install_gepa_stubs() -> None:
    from rg_tracer.testing.gepa_stubs import install_gepa_stubs

    install_gepa_stubs()


_install_gepa_stubs()


def _install_gepa_stubs() -> None:
    from rg_tracer.testing.gepa_stubs import install_gepa_stubs

    install_gepa_stubs()


_install_gepa_stubs()


def _install_gepa_stubs() -> None:
    from rg_tracer.testing.gepa_stubs import install_gepa_stubs

    install_gepa_stubs()


_install_gepa_stubs()


def _install_gepa_stubs() -> None:
    try:
        import gepa_dapo_grn  # noqa: F401
    except ImportError:
        from rg_tracer.testing.gepa_stubs import install_gepa_stubs

        install_gepa_stubs()


_install_gepa_stubs()


def _install_gepa_stubs() -> None:
    if importlib.util.find_spec("gepa_dapo_grn") is None:
        from rg_tracer.testing.gepa_stubs import install_gepa_stubs

        install_gepa_stubs()


_install_gepa_stubs()


def _install_gepa_stubs() -> None:
    if importlib.util.find_spec("gepa_dapo_grn") is None:
        from rg_tracer.testing.gepa_stubs import install_gepa_stubs

        install_gepa_stubs()


_install_gepa_stubs()


@pytest.fixture(autouse=True)
def reset_aggregator_defaults():
    from rg_tracer.scoring import aggregator

    with aggregator._LAST_CONFIG_LOCK:
        prior_last_config = copy.deepcopy(aggregator._LAST_CONFIG)
        aggregator._LAST_CONFIG = copy.deepcopy(aggregator.DEFAULT_CONFIG)
    yield
    with aggregator._LAST_CONFIG_LOCK:
        aggregator._LAST_CONFIG = prior_last_config
