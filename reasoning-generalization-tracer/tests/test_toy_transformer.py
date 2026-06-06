"""Tests for the optional toy transformer."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from rg_tracer.grokking.config import ToyGrokkingConfig
from rg_tracer.grokking.toy_transformer import (
    ToyGrokkingTransformer,
    make_modular_addition_dataset,
)


def test_fixed_seed_deterministic_dataset():
    config = ToyGrokkingConfig(modulus=7, seed=3)
    left = make_modular_addition_dataset(config)
    right = make_modular_addition_dataset(config)
    assert left.train_inputs.tolist() == right.train_inputs.tolist()


def test_toy_transformer_forward_shape():
    config = ToyGrokkingConfig(modulus=7, d_model=16, n_heads=2)
    dataset = make_modular_addition_dataset(config)
    model = ToyGrokkingTransformer(config)
    logits = model(dataset.train_inputs[:4])
    assert list(logits.shape) == [4, 7]
