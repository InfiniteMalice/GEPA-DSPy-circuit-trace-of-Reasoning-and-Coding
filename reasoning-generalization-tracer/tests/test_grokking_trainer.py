"""Smoke tests for the optional grokking trainer."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from rg_tracer.grokking.config import ToyGrokkingConfig
from rg_tracer.grokking.trainer import train_toy_grokking


def test_training_smoke_writes_artifacts(tmp_path: Path):
    config = ToyGrokkingConfig(
        modulus=5,
        epochs=2,
        eval_interval=1,
        batch_size=16,
        d_model=8,
        n_heads=2,
    )
    result = train_toy_grokking(config, output_dir=tmp_path)
    assert result.summary["final_train_accuracy"] is not None
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "spectra.jsonl").exists()
    assert (tmp_path / "config.json").exists()
    assert result.metrics[-1]["memorization_epoch"] is None
