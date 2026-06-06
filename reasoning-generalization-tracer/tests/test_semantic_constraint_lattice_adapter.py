"""Tests for semantic constraints routed into finite lattices."""

from __future__ import annotations

from rg_tracer.recursive_refinement.lattice_adapters import select_lattice_adapter
from rg_tracer.semantic_constraints.adapters import compile_verified_constraints_to_lattice


def test_gated_toy_only_resolves_unique_candidate():
    projection = compile_verified_constraints_to_lattice(
        "The answer must be even and greater than 2.",
        [1, 2, 3, 4],
        mode="gated_toy_only",
    )
    assert projection.gated
    assert projection.remaining_candidates == [4]


def test_shadow_mode_never_gates():
    projection = compile_verified_constraints_to_lattice(
        "The answer must be even.",
        [1, 2, 3, 4],
        mode="shadow",
    )
    assert not projection.gated


def test_open_ended_task_cannot_gate():
    projection = compile_verified_constraints_to_lattice(
        "The answer must be even.",
        [1, 2, 3, 4],
        mode="gated_toy_only",
        task_type="open_ended",
    )
    assert not projection.gated


def test_lattice_adapter_compiles_verified_constraints():
    problem = {
        "task": "semantic_constraint_toy",
        "domain": [1, 2, 3, 4],
        "requirement": "The answer must be even and greater than 2.",
    }
    adapter = select_lattice_adapter(problem)
    assert adapter is not None
    constraints = adapter.constraints(problem)
    allowed_sets = [set(constraint.allowed_candidates) for constraint in constraints]
    combined_allowed = frozenset(set.intersection(*allowed_sets))
    assert combined_allowed == frozenset({4})
