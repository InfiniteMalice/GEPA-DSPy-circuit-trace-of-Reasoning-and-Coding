"""Tests for PTRM and lattice sampler ladder branches."""

from __future__ import annotations

from rg_tracer.recursive_refinement import (
    LatticeConfig,
    LatticePTRMSampler,
    LatticeTRMSampler,
    PTRMSampler,
    PerturbationConfig,
    RecursiveRefinementConfig,
    RefinementBudget,
)


def test_ptrm_sampler_is_seed_deterministic_and_bounded_width():
    config = RecursiveRefinementConfig(
        budget=RefinementBudget(max_depth=3, max_width=4, seed=3),
        perturbation=PerturbationConfig(enabled=True, trajectories=3, seed=11),
    )
    problem = {"id": "p", "task": "addition", "numbers": [2, 3], "answer": 5}
    first = PTRMSampler(config).generate(problem, k=3)
    second = PTRMSampler(config).generate(problem, k=3)
    assert [item["prediction"] for item in first] == [item["prediction"] for item in second]
    assert len(first) == 3
    assert all(item["perturbations"] for item in first)


def test_lattice_trm_gated_resolves_singleton_candidate():
    config = RecursiveRefinementConfig(lattice=LatticeConfig(mode="gated"))
    problem = {"id": "p", "task": "addition", "numbers": [2, 3], "answer": 5}
    candidate = LatticeTRMSampler(config).generate(problem, k=1)[0]
    assert candidate["prediction"] == 5
    assert candidate["lattice_diagnostics"]["resolved"]


def test_lattice_ptrm_gated_attaches_projection_metadata():
    config = RecursiveRefinementConfig(
        budget=RefinementBudget(max_depth=2, max_width=3, seed=7),
        lattice=LatticeConfig(mode="gated"),
        perturbation=PerturbationConfig(enabled=True, trajectories=3, seed=7),
    )
    problem = {
        "id": "fdc",
        "task": "finite_domain_constraint",
        "domain": [1, 2, 3, 4],
        "constraints": [
            {"id": "even", "allowed": [2, 4]},
            {"id": "gt2", "allowed": [3, 4]},
        ],
        "answer": 4,
    }
    candidate = LatticePTRMSampler(config).generate(problem, k=2)[0]
    assert candidate["prediction"] == 4
    assert candidate["lattice_diagnostics"]["resolved"]
