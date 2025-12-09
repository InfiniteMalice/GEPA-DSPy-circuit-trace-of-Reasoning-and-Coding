"""Confidence-aware abstention policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..modules.grn import apply_grn
from ..modules.torch_stub import torch

ABSTENTION_THRESHOLD = 0.75
SEMANTIC_THRESHOLD = 2


@dataclass
class AbstentionResult:
    text: str
    abstained: bool
    confidence: float


def apply_abstention(
    output_text: str,
    confidence: float,
    sem_score: float,
    gates_pass: bool = True,
    *,
    use_grn: bool = False,
    grn_eps: float = 1e-6,
) -> AbstentionResult:
    """Return abstention result enforcing the 0.75 threshold and semantic gate."""
    vector = torch.tensor([confidence, sem_score], dtype=torch.float32)
    if use_grn:
        vector = apply_grn(vector, eps=grn_eps)
    confidence_value = float(vector[0].item())
    sem_value = float(vector[1].item())
    should_abstain = (
        confidence_value < ABSTENTION_THRESHOLD or sem_value < SEMANTIC_THRESHOLD or not gates_pass
    )
    if should_abstain:
        return AbstentionResult(text="I don't know.", abstained=True, confidence=confidence_value)
    return AbstentionResult(text=output_text, abstained=False, confidence=confidence_value)


def apply_abstention_tuple(
    output_text: str,
    confidence: float,
    sem_score: float,
    gates_pass: bool = True,
    *,
    use_grn: bool = False,
    grn_eps: float = 1e-6,
) -> Tuple[str, bool]:
    result = apply_abstention(
        output_text,
        confidence,
        sem_score,
        gates_pass,
        use_grn=use_grn,
        grn_eps=grn_eps,
    )
    return result.text, result.abstained


__all__ = [
    "apply_abstention",
    "apply_abstention_tuple",
    "AbstentionResult",
    "ABSTENTION_THRESHOLD",
    "SEMANTIC_THRESHOLD",
]
