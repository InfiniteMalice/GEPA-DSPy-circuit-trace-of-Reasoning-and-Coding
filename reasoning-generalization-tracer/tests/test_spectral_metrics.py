"""Tests for real spectral metrics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rg_tracer.grokking.spectral_metrics import effective_rank, rank_collapse_ratio, stable_rank


def test_effective_rank_rank_one_lower_than_full_rank():
    rank_one = torch.ones(4, 4)
    full_rank = torch.eye(4)
    assert effective_rank(rank_one) < effective_rank(full_rank)


def test_stable_rank_simple_diagonal():
    matrix = torch.diag(torch.tensor([3.0, 4.0]))
    assert stable_rank(matrix) == pytest.approx((9.0 + 16.0) / 16.0)


def test_rank_collapse_ratio_uses_actual_matrices():
    initial = torch.eye(4)
    current = torch.ones(4, 4)
    assert rank_collapse_ratio(initial, current) > 0.0
