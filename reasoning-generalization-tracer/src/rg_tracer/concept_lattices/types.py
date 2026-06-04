"""Types for explicit human-authored conceptual lattice examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConceptAttribute:
    """A public conceptual attribute for diagnostic examples."""

    name: str
    description: str


@dataclass(frozen=True)
class ConceptLatticeSpec:
    """A tiny explicit concept lattice specification for shadow evaluation."""

    name: str
    domain: str
    attributes: tuple[ConceptAttribute, ...]
    implications: tuple[tuple[str, str], ...] = ()
    shadow_only: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConceptLatticeExample:
    """Synthetic diagnostic example, never an external-action controller."""

    example_id: str
    prompt: str
    expected_attributes: tuple[str, ...]
    lattice_name: str
    notes: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["ConceptAttribute", "ConceptLatticeExample", "ConceptLatticeSpec"]
