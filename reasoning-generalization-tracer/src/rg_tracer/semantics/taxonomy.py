"""Semantic error taxonomy used by the verifier and repair routines."""

from __future__ import annotations

from enum import Enum


class SemanticTag(str, Enum):
    CONTRADICTION = "CONTRADICTION"
    UNSUPPORTED = "UNSUPPORTED"
    VARIABLE_DRIFT = "VARIABLE_DRIFT"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    DEFINITION_DRIFT = "DEFINITION_DRIFT"
    ILLEGAL_INFERENCE = "ILLEGAL_INFERENCE"
    CIRCULARITY = "CIRCULARITY"
    ENTAILED = "ENTAILED"
    SUPPORTED = "SUPPORTED"


__all__ = ["SemanticTag"]
