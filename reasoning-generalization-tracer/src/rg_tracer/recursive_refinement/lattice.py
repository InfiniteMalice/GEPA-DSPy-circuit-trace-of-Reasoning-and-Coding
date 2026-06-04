"""Explicit finite candidate lattices for bounded toy reasoning tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _stable_items(values: frozenset[object]) -> list[object]:
    return sorted(values, key=lambda item: (type(item).__name__, repr(item)))


@dataclass(frozen=True)
class LatticeElement:
    """A finite lattice element represented by a set of candidate answers."""

    element_id: str
    candidates: frozenset[object]
    attributes: frozenset[str] = frozenset()

    def as_dict(self) -> dict[str, object]:
        return {
            "element_id": self.element_id,
            "candidates": _stable_items(self.candidates),
            "attributes": sorted(self.attributes),
        }


@dataclass(frozen=True)
class DeductionConstraint:
    """An explicit candidate-elimination constraint."""

    constraint_id: str
    description: str
    allowed_candidates: frozenset[object]
    source: str = "explicit"

    def as_dict(self) -> dict[str, object]:
        return {
            "constraint_id": self.constraint_id,
            "description": self.description,
            "allowed_candidates": _stable_items(self.allowed_candidates),
            "source": self.source,
        }


@dataclass
class ProjectionStep:
    """Public record for one lattice projection operation."""

    step_index: int
    constraint_id: str
    before_candidates: list[object]
    after_candidates: list[object]
    removed_candidates: list[object]
    operation: str
    contradiction: bool
    resolved: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LatticeProjectionResult:
    """JSON-safe projection summary for one task-local lattice."""

    initial_candidates: list[object]
    remaining_candidates: list[object]
    projection_steps: list[ProjectionStep]
    meet_count: int
    join_count: int
    contradiction_detected: bool
    resolved: bool
    unresolved: bool
    abstain_recommended: bool
    abstain_reason: str | None
    adapter_name: str | None = None
    mode: str = "off"
    canonicalization_count: int = 0
    merged_branch_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["projection_steps"] = [step.as_dict() for step in self.projection_steps]
        return payload


class FiniteCandidateLattice:
    """Finite lattice over candidate sets using subset inclusion as the order."""

    def __init__(self) -> None:
        self._canonical: dict[tuple[str, ...], LatticeElement] = {}

    def element(
        self,
        candidates: list[object] | frozenset[object],
        *,
        attributes: frozenset[str] = frozenset(),
    ) -> LatticeElement:
        return self.canonicalize(
            LatticeElement(
                element_id=self._element_id(frozenset(candidates)),
                candidates=frozenset(candidates),
                attributes=attributes,
            )
        )

    def meet(self, left: LatticeElement, right: LatticeElement) -> LatticeElement:
        return self.canonicalize(
            LatticeElement(
                element_id=self._element_id(left.candidates & right.candidates),
                candidates=left.candidates & right.candidates,
                attributes=left.attributes | right.attributes | frozenset({"meet"}),
            )
        )

    def join(self, left: LatticeElement, right: LatticeElement) -> LatticeElement:
        return self.canonicalize(
            LatticeElement(
                element_id=self._element_id(left.candidates | right.candidates),
                candidates=left.candidates | right.candidates,
                attributes=left.attributes | right.attributes | frozenset({"join"}),
            )
        )

    def is_below(self, left: LatticeElement, right: LatticeElement) -> bool:
        return left.candidates <= right.candidates

    def canonicalize(self, element: LatticeElement) -> LatticeElement:
        key = tuple(repr(item) for item in _stable_items(element.candidates))
        existing = self._canonical.get(key)
        if existing is not None:
            return existing
        canonical = LatticeElement(
            element_id=self._element_id(element.candidates),
            candidates=frozenset(_stable_items(element.candidates)),
            attributes=frozenset(sorted(element.attributes)),
        )
        self._canonical[key] = canonical
        return canonical

    def project(
        self,
        element: LatticeElement,
        constraint: DeductionConstraint,
    ) -> LatticeElement:
        constraint_element = self.element(
            constraint.allowed_candidates,
            attributes=frozenset({constraint.constraint_id}),
        )
        return self.meet(element, constraint_element)

    def _element_id(self, candidates: frozenset[object]) -> str:
        if not candidates:
            return "bottom"
        return "cand:" + ",".join(repr(item) for item in _stable_items(candidates))


__all__ = [
    "DeductionConstraint",
    "FiniteCandidateLattice",
    "LatticeElement",
    "LatticeProjectionResult",
    "ProjectionStep",
]
