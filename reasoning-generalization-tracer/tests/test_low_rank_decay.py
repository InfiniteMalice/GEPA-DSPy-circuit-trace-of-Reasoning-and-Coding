"""Tests for LRD-inspired regularization."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rg_tracer.grokking.config import SpectralRegularizationConfig
from rg_tracer.grokking.regularizers import (
    apply_low_rank_decay_,
    polar_factor_exact,
    polar_factor_newton_schulz,
)


def _params():
    return {
        "layers.0.attn.q_proj.weight": torch.nn.Parameter(torch.eye(3)),
        "layers.0.attn.k_proj.weight": torch.nn.Parameter(torch.eye(3) * 2.0),
        "layers.0.attn.v_proj.weight": torch.nn.Parameter(torch.eye(3) * 3.0),
    }


def test_exact_polar_factor_shape():
    matrix = torch.randn(3, 2)
    assert polar_factor_exact(matrix).shape == matrix.shape


def test_newton_schulz_polar_factor_is_finite():
    matrix = torch.randn(3, 3)
    approx = polar_factor_newton_schulz(matrix, steps=3)
    assert torch.isfinite(approx).all()


def test_lrd_update_changes_only_selected_matrices():
    params = _params()
    before_q = params["layers.0.attn.q_proj.weight"].detach().clone()
    before_v = params["layers.0.attn.v_proj.weight"].detach().clone()
    config = SpectralRegularizationConfig(mode="lrd", lrd_weight=0.01)
    diagnostics = apply_low_rank_decay_(params.items(), config)
    assert diagnostics
    assert not torch.equal(params["layers.0.attn.q_proj.weight"], before_q)
    assert torch.equal(params["layers.0.attn.v_proj.weight"], before_v)


def test_regularization_modes():
    for mode, kwargs in (
        ("l2", {"l2_weight": 0.01}),
        ("lrd", {"lrd_weight": 0.01}),
        ("l2_plus_lrd", {"l2_weight": 0.01, "lrd_weight": 0.01}),
    ):
        params = _params()
        before = params["layers.0.attn.q_proj.weight"].detach().clone()
        config = SpectralRegularizationConfig(mode=mode, **kwargs)
        apply_low_rank_decay_(params.items(), config)
        assert not torch.equal(params["layers.0.attn.q_proj.weight"], before)


def test_mode_none_leaves_weights_unchanged():
    params = _params()
    before = params["layers.0.attn.q_proj.weight"].detach().clone()
    diagnostics = apply_low_rank_decay_(params.items(), SpectralRegularizationConfig())
    assert diagnostics == []
    assert torch.equal(params["layers.0.attn.q_proj.weight"], before)
