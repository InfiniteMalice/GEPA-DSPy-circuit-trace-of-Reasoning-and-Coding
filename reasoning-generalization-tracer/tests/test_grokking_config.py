"""Tests for optional grokking configuration and base imports."""

from __future__ import annotations

import pytest

from rg_tracer.grokking import SpectralRegularizationConfig, ToyGrokkingConfig


def test_base_grokking_imports_without_torch():
    config = ToyGrokkingConfig(modulus=7, epochs=5, eval_interval=1)
    regularization = SpectralRegularizationConfig(mode="none")
    assert config.modulus == 7
    assert regularization.mode == "none"


def test_invalid_regularization_mode_fails_early():
    with pytest.raises(ValueError, match="mode"):
        SpectralRegularizationConfig(mode="bad")


def test_lrd_mode_requires_positive_weight():
    with pytest.raises(ValueError, match="lrd_weight"):
        SpectralRegularizationConfig(mode="lrd", lrd_weight=0.0)


def test_torch_module_fails_clearly_when_absent():
    from rg_tracer.grokking.spectral_metrics import require_torch

    try:
        require_torch()
    except ImportError as exc:
        assert "grokking" in str(exc)
