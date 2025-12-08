"""Lightweight shim to provide torch-like APIs when PyTorch is unavailable."""

from __future__ import annotations

import importlib.util
from typing import Any, Iterable

if importlib.util.find_spec("torch"):
    import torch  # type: ignore

    SimpleTensor = torch.Tensor  # type: ignore
else:  # pragma: no cover - fallback path
    _SCALAR_INDEXING_ERROR = "SimpleTensor does not support indexing on scalars"
    _SCALAR_ITERATION_ERROR = "SimpleTensor is not iterable"

    class SimpleTensor:
        def __init__(self, data: Any):
            self.data = data

        def dim(self) -> int:
            return self.ndim

        @property
        def ndim(self) -> int:
            if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
                return 1 + SimpleTensor(self.data[0]).ndim
            if isinstance(self.data, list):
                return 1
            return 0

        def tolist(self) -> Any:
            if isinstance(self.data, list):
                return [SimpleTensor(item).tolist() for item in self.data]
            return self.data

        def item(self) -> float:
            if isinstance(self.data, list):
                if len(self.data) != 1:
                    raise ValueError("only one element tensors can be converted to Python scalars")
                return SimpleTensor(self.data[0]).item()
            return float(self.data)

        def __getitem__(self, index) -> "SimpleTensor":
            if isinstance(self.data, list):
                return SimpleTensor(self.data[index])
            raise TypeError(_SCALAR_INDEXING_ERROR)

        def __iter__(self):
            if isinstance(self.data, list):
                return (SimpleTensor(item) for item in self.data)
            raise TypeError(_SCALAR_ITERATION_ERROR)

    class _TorchShim:
        float32 = float

        def tensor(self, data: Any, dtype: object | None = None) -> SimpleTensor:  # noqa: ARG002
            if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
                return SimpleTensor([self._coerce(item) for item in data])
            return SimpleTensor(self._coerce(data))

        def _coerce(self, value: Any) -> Any:
            if isinstance(value, SimpleTensor):
                return value.data
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                return [self._coerce(item) for item in value]
            return float(value)

    torch = _TorchShim()

# ``SimpleTensor`` aliases ``torch.Tensor`` when the real library is available; otherwise the
# shim-defined SimpleTensor is exported for compatibility so ``__all__`` is stable across
# environments.
__all__ = ["SimpleTensor", "torch"]
