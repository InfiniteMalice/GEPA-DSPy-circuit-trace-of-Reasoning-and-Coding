"""Tests for PTRM-inspired bounded Gaussian perturbations."""

from __future__ import annotations

from rg_tracer.recursive_refinement import PerturbationConfig
from rg_tracer.recursive_refinement.perturbation import GaussianTrajectoryPerturber
from rg_tracer.recursive_refinement.types import RefinementState


def _state() -> RefinementState:
    return RefinementState(
        trajectory_id="t",
        parent_id=None,
        depth=0,
        state_value=0.2,
        uncertainty=0.3,
        confidence=0.5,
        prediction=1,
        active_views=[],
        operations=[],
    )


def test_fixed_seed_is_reproducible_and_bounded():
    config = PerturbationConfig(enabled=True, seed=7, noise_std=0.5)
    first = GaussianTrajectoryPerturber(config).maybe_perturb(
        _state(), trajectory_id="a", parent_id=None, depth=1
    )[1]
    second = GaussianTrajectoryPerturber(config).maybe_perturb(
        _state(), trajectory_id="a", parent_id=None, depth=1
    )[1]
    assert first is not None
    assert second is not None
    assert first.bounded_noise == second.bounded_noise
    assert -0.25 <= first.bounded_noise <= 0.25


def test_perturbation_starts_after_configured_depth():
    config = PerturbationConfig(enabled=True, perturb_after_depth=2, seed=7)
    perturber = GaussianTrajectoryPerturber(config)
    assert perturber.maybe_perturb(_state(), trajectory_id="a", parent_id=None, depth=1)[1] is None
    assert perturber.maybe_perturb(_state(), trajectory_id="a", parent_id=None, depth=2)[1]
