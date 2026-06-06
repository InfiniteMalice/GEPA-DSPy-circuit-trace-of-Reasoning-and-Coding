"""CPU-friendly modular-addition transformer for real toy grokking runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ToyGrokkingConfig
from .spectral_metrics import require_torch

try:  # pragma: no cover - import guard is validated by tests without torch.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ModularAdditionDataset:
    """Train/test tensors for modular addition."""

    train_inputs: Any
    train_labels: Any
    test_inputs: Any
    test_labels: Any


def make_modular_addition_dataset(config: ToyGrokkingConfig) -> ModularAdditionDataset:
    """Generate a deterministic modular-addition dataset."""

    torch_mod = require_torch()
    pairs = []
    labels = []
    for left in range(config.modulus):
        for right in range(config.modulus):
            pairs.append([left, right])
            labels.append((left + right) % config.modulus)
    inputs = torch_mod.tensor(pairs, dtype=torch_mod.long)
    targets = torch_mod.tensor(labels, dtype=torch_mod.long)
    generator = torch_mod.Generator().manual_seed(config.seed)
    order = torch_mod.randperm(len(inputs), generator=generator)
    split = max(1, min(len(inputs) - 1, int(len(inputs) * config.train_fraction)))
    train_idx = order[:split]
    test_idx = order[split:]
    return ModularAdditionDataset(
        train_inputs=inputs[train_idx],
        train_labels=targets[train_idx],
        test_inputs=inputs[test_idx],
        test_labels=targets[test_idx],
    )


if nn is not None:

    class RMSNorm(nn.Module):
        """Small RMSNorm approximation for scale-stabilized toy runs."""

        def __init__(self, d_model: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d_model))
            self.eps = eps

        def forward(self, values: Any) -> Any:
            rms = values.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            return self.weight * values / rms

    class ToySelfAttention(nn.Module):
        """Minimal attention block with explicit Q/K/V projection matrices."""

        def __init__(self, d_model: int, n_heads: int) -> None:
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, values: Any) -> Any:
            batch, seq_len, _ = values.shape
            q = self._split_heads(self.q_proj(values), batch, seq_len)
            k = self._split_heads(self.k_proj(values), batch, seq_len)
            v = self._split_heads(self.v_proj(values), batch, seq_len)
            scale = float(self.head_dim) ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            probs = scores.softmax(dim=-1)
            attended = torch.matmul(probs, v)
            merged = attended.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
            return self.out_proj(merged)

        def _split_heads(self, values: Any, batch: int, seq_len: int) -> Any:
            values = values.view(batch, seq_len, self.n_heads, self.head_dim)
            return values.transpose(1, 2)

    class ToyTransformerBlock(nn.Module):
        """One compact transformer block for modular addition."""

        def __init__(self, d_model: int, n_heads: int) -> None:
            super().__init__()
            self.norm_1 = RMSNorm(d_model)
            self.attn = ToySelfAttention(d_model, n_heads)
            self.norm_2 = RMSNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model),
            )

        def forward(self, values: Any) -> Any:
            values = values + self.attn(self.norm_1(values))
            return values + self.mlp(self.norm_2(values))

    class ToyGrokkingTransformer(nn.Module):
        """Tiny transformer over two-token modular-addition inputs."""

        def __init__(self, config: ToyGrokkingConfig) -> None:
            super().__init__()
            torch.manual_seed(config.seed)
            self.config = config
            self.token_embedding = nn.Embedding(config.modulus, config.d_model)
            self.position_embedding = nn.Parameter(torch.zeros(2, config.d_model))
            self.layers = nn.ModuleList(
                [
                    ToyTransformerBlock(config.d_model, config.n_heads)
                    for _ in range(config.n_layers)
                ]
            )
            self.norm = RMSNorm(config.d_model)
            self.classifier = nn.Linear(config.d_model, config.modulus)

        def forward(self, inputs: Any) -> Any:
            values = self.token_embedding(inputs) + self.position_embedding
            for layer in self.layers:
                values = layer(values)
            pooled = self.norm(values).mean(dim=1)
            return self.classifier(pooled)

else:

    class ToyGrokkingTransformer:  # type: ignore[no-redef]
        """Placeholder that raises when Torch is not installed."""

        def __init__(self, config: ToyGrokkingConfig) -> None:
            require_torch()


__all__ = [
    "ModularAdditionDataset",
    "ToyGrokkingTransformer",
    "make_modular_addition_dataset",
]
