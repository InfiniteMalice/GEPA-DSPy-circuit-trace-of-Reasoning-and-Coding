"""Shadow-only interfaces for future representation-lattice probes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol


class RepresentationLatticeProbe(Protocol):
    """Interface for future LRH-inspired probes; no fake recovery is implemented."""

    def extract_attribute_scores(
        self,
        activations: object,
        attributes: Sequence[str],
    ) -> Mapping[str, float]: ...

    def estimate_thresholds(
        self,
        scores: Mapping[str, Sequence[float]],
    ) -> Mapping[str, float]: ...

    def build_shadow_lattice(
        self,
        scores: Mapping[str, float],
        thresholds: Mapping[str, float],
    ) -> Mapping[str, object]: ...

    def compare_against_reference(
        self,
        shadow_lattice: Mapping[str, object],
        reference_lattice: Mapping[str, object],
    ) -> Mapping[str, float]: ...


__all__ = ["RepresentationLatticeProbe"]
