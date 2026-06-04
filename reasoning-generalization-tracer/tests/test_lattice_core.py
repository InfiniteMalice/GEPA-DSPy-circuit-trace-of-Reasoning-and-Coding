"""Tests for explicit finite candidate lattice operations."""

from __future__ import annotations

from rg_tracer.recursive_refinement.lattice import (
    DeductionConstraint,
    FiniteCandidateLattice,
)


def test_meet_join_and_order_use_candidate_sets():
    lattice = FiniteCandidateLattice()
    left = lattice.element([1, 2, 3])
    right = lattice.element([2, 3, 4])
    meet = lattice.meet(left, right)
    join = lattice.join(left, right)
    assert meet.candidates == frozenset({2, 3})
    assert join.candidates == frozenset({1, 2, 3, 4})
    assert lattice.is_below(meet, left)
    assert not lattice.is_below(join, left)


def test_projection_and_canonicalization_are_deterministic():
    lattice = FiniteCandidateLattice()
    element = lattice.element([3, 1, 2])
    same = lattice.element([2, 1, 3])
    projected = lattice.project(
        element,
        DeductionConstraint("only_two", "Only two remains.", frozenset({2})),
    )
    assert same is element
    assert projected.candidates == frozenset({2})
    assert projected.as_dict()["candidates"] == [2]
