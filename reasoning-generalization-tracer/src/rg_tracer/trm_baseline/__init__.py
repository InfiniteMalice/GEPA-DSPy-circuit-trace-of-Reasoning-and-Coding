"""Tiny Recursion Model baseline."""

from .trm_model import TinyRecursionModel
from .trm_train import (
    quickstart_trainer,
    train,
    generate_parity_data,
    generate_carry_data,
)
from .trm_eval import evaluate

__all__ = [
    "TinyRecursionModel",
    "evaluate",
    "generate_carry_data",
    "generate_parity_data",
    "quickstart_trainer",
    "train",
]
