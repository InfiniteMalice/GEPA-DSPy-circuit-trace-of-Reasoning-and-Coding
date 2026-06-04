"""Task-local adapters that expose explicit finite candidate lattices."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from .lattice import DeductionConstraint


class LatticeAdapter(Protocol):
    """Adapter from a public toy problem to candidates and explicit constraints."""

    name: str

    def applicable(self, problem: Mapping[str, object]) -> bool: ...

    def initial_candidates(self, problem: Mapping[str, object]) -> list[object]: ...

    def constraints(self, problem: Mapping[str, object]) -> list[DeductionConstraint]: ...


def _ints(value: object) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, int) and not isinstance(item, bool)]


class AdditionLatticeAdapter:
    name = "addition"

    def applicable(self, problem: Mapping[str, object]) -> bool:
        return problem.get("task") == "addition" and bool(_ints(problem.get("numbers")))

    def initial_candidates(self, problem: Mapping[str, object]) -> list[object]:
        numbers = _ints(problem.get("numbers"))
        total = sum(numbers)
        spread = max(2, len(numbers))
        return list(range(total - spread, total + spread + 1))

    def constraints(self, problem: Mapping[str, object]) -> list[DeductionConstraint]:
        total = sum(_ints(problem.get("numbers")))
        return [
            DeductionConstraint(
                constraint_id="sum_public_numbers",
                description="Candidate must equal the sum of the public numbers.",
                allowed_candidates=frozenset({total}),
            )
        ]


class ParityLatticeAdapter:
    name = "parity"

    def applicable(self, problem: Mapping[str, object]) -> bool:
        return problem.get("task") == "parity" and bool(_ints(problem.get("sequence")))

    def initial_candidates(self, problem: Mapping[str, object]) -> list[object]:
        return [0, 1]

    def constraints(self, problem: Mapping[str, object]) -> list[DeductionConstraint]:
        parity = sum(_ints(problem.get("sequence"))) % 2
        return [
            DeductionConstraint(
                constraint_id="sequence_parity",
                description="Candidate must match the parity of the public sequence.",
                allowed_candidates=frozenset({parity}),
            )
        ]


class FiniteDomainConstraintAdapter:
    name = "finite_domain_constraint"

    def applicable(self, problem: Mapping[str, object]) -> bool:
        return problem.get("task") == "finite_domain_constraint" and isinstance(
            problem.get("domain"), list
        )

    def initial_candidates(self, problem: Mapping[str, object]) -> list[object]:
        domain = problem.get("domain")
        return list(domain) if isinstance(domain, list) else []

    def constraints(self, problem: Mapping[str, object]) -> list[DeductionConstraint]:
        constraints = problem.get("constraints", [])
        if not isinstance(constraints, list):
            return []
        parsed = []
        for index, raw in enumerate(constraints):
            if not isinstance(raw, Mapping):
                continue
            allowed = raw.get("allowed")
            if not isinstance(allowed, list):
                continue
            constraint_id = str(raw.get("id") or f"constraint_{index}")
            parsed.append(
                DeductionConstraint(
                    constraint_id=constraint_id,
                    description=str(raw.get("description") or constraint_id),
                    allowed_candidates=frozenset(allowed),
                    source=str(raw.get("source") or "explicit"),
                )
            )
        return parsed


def available_lattice_adapters() -> dict[str, LatticeAdapter]:
    adapters: list[LatticeAdapter] = [
        AdditionLatticeAdapter(),
        ParityLatticeAdapter(),
        FiniteDomainConstraintAdapter(),
    ]
    return {adapter.name: adapter for adapter in adapters}


def select_lattice_adapter(
    problem: Mapping[str, object],
    adapter_name: str = "auto",
) -> LatticeAdapter | None:
    adapters = available_lattice_adapters()
    if adapter_name != "auto":
        adapter = adapters.get(adapter_name)
        if adapter is None or not adapter.applicable(problem):
            return None
        return adapter
    for adapter in adapters.values():
        if adapter.applicable(problem):
            return adapter
    return None


__all__ = [
    "AdditionLatticeAdapter",
    "FiniteDomainConstraintAdapter",
    "LatticeAdapter",
    "ParityLatticeAdapter",
    "available_lattice_adapters",
    "select_lattice_adapter",
]
