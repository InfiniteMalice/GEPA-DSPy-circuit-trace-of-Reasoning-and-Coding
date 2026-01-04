"""Tiny Recursion Model (TRM) implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


@dataclass
class TinyRecursionModel:
    """A lightweight recursive model for binary sequence reasoning."""

    hidden_scale: float = 0.5
    recurse_scale: float = 0.5
    bias: float = 0.0
    learning_rate: float = 0.1
    trace_states: bool = True
    history: List[Tuple[List[int], List[float]]] = field(default_factory=list)

    def _recurse(self, seq: Sequence[int]) -> Tuple[float, List[float]]:
        if not seq:
            return 0.0, [0.0]
        head, *rest = seq
        rest_value, rest_trace = self._recurse(rest)
        activation = math.tanh(
            self.hidden_scale * head + self.recurse_scale * rest_value + self.bias
        )
        if self.trace_states:
            return activation, [activation] + rest_trace
        return activation, []

    def forward(self, seq: Sequence[int]) -> Tuple[float, List[float]]:
        return self._recurse(list(seq))

    def predict(self, seq: Sequence[int]) -> float:
        value, _ = self.forward(seq)
        # Map tanh output [-1,1] to [0,1]
        return (value + 1) / 2

    def _gradients(
        self, seq: Sequence[int], target: float
    ) -> Tuple[float, float, float]:
        if not seq:
            pred = self.predict(seq)
            error = pred - target
            return error, 0.0, 0.0
        head, *rest = seq
        rest_value, rest_trace = self._recurse(rest)
        activation = math.tanh(
            self.hidden_scale * head + self.recurse_scale * rest_value + self.bias
        )
        pred = (activation + 1) / 2
        error = pred - target
        sech2 = 1 - activation**2
        d_hidden = error * 0.5 * sech2 * head
        d_recurse = error * 0.5 * sech2 * rest_value
        d_bias = error * 0.5 * sech2
        return d_hidden, d_recurse, d_bias

    def train_step(self, seq: Sequence[int], target: float) -> float:
        d_hidden, d_recurse, d_bias = self._gradients(seq, target)
        self.hidden_scale -= self.learning_rate * d_hidden
        self.recurse_scale -= self.learning_rate * d_recurse
        self.bias -= self.learning_rate * d_bias
        # Clamp parameters to stabilise training on toy tasks
        self.hidden_scale = max(min(self.hidden_scale, 2.0), -2.0)
        self.recurse_scale = max(min(self.recurse_scale, 2.0), -2.0)
        self.bias = max(min(self.bias, 2.0), -2.0)
        pred = self.predict(seq)
        loss = 0.5 * (pred - target) ** 2
        if self.trace_states:
            activation, trace = self.forward(seq)
            self.history.append((list(seq), trace))
        return loss

    def reset_history(self) -> None:
        self.history.clear()


__all__ = ["TinyRecursionModel"]
