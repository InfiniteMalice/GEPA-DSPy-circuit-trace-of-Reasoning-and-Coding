"""Tests for task-local lattice adapters."""

from __future__ import annotations

from rg_tracer.recursive_refinement.lattice_adapters import select_lattice_adapter


def test_addition_adapter_resolves_sum_constraint():
    problem = {"task": "addition", "numbers": [2, 3], "answer": 5}
    adapter = select_lattice_adapter(problem)
    assert adapter is not None
    assert adapter.name == "addition"
    assert 5 in adapter.initial_candidates(problem)
    assert adapter.constraints(problem)[0].allowed_candidates == frozenset({5})


def test_finite_domain_adapter_reads_explicit_constraints():
    problem = {
        "task": "finite_domain_constraint",
        "domain": [1, 2, 3, 4],
        "constraints": [{"id": "even", "allowed": [2, 4]}],
    }
    adapter = select_lattice_adapter(problem)
    assert adapter is not None
    assert adapter.initial_candidates(problem) == [1, 2, 3, 4]
    assert adapter.constraints(problem)[0].constraint_id == "even"
