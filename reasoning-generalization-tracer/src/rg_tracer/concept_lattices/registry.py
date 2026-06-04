"""Registry for shadow-only explicit conceptual lattice examples."""

from __future__ import annotations

from .types import ConceptLatticeSpec


class ConceptLatticeRegistry:
    """In-memory registry for human-authored diagnostic concept lattices."""

    def __init__(self) -> None:
        self._items: dict[str, ConceptLatticeSpec] = {}

    def register(self, spec: ConceptLatticeSpec) -> None:
        if not spec.shadow_only:
            raise ValueError("concept lattice specs are shadow-only in this scaffold")
        if spec.name in self._items:
            raise ValueError(f"duplicate concept lattice name: {spec.name}")
        self._items[spec.name] = spec

    def get(self, name: str) -> ConceptLatticeSpec | None:
        return self._items.get(name)

    def names(self) -> list[str]:
        return sorted(self._items)


__all__ = ["ConceptLatticeRegistry"]
