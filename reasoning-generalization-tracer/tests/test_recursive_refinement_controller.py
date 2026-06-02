"""Tests for the adaptive recursive-refinement controller."""

from __future__ import annotations

from rg_tracer.recursive_refinement import (
    RecursiveRefinementConfig,
    RecursiveRefinementController,
    RefinementBudget,
    ViewRoutingConfig,
)


def _problem():
    return {"id": "p", "task": "addition", "numbers": [2, 3], "answer": 5}


def test_controller_widens_when_uncertainty_triggers_it():
    config = RecursiveRefinementConfig(
        budget=RefinementBudget(max_depth=4, max_width=4, max_total_updates=24, seed=7)
    )
    run = RecursiveRefinementController(config).run(_problem(), target_width=4)
    assert len(run.trajectories) > 1
    assert run.max_observed_width <= 4


def test_controller_is_deterministic_for_fixed_seed():
    config = RecursiveRefinementConfig(
        budget=RefinementBudget(max_depth=5, max_width=4, max_total_updates=24, seed=3)
    )
    first = RecursiveRefinementController(config).run(_problem(), target_width=4).as_dict()
    second = RecursiveRefinementController(config).run(_problem(), target_width=4).as_dict()
    assert first == second


def test_different_seeds_can_produce_different_valid_routes():
    first_config = RecursiveRefinementConfig(
        budget=RefinementBudget(max_depth=5, max_width=4, max_total_updates=24, seed=1)
    )
    second_config = RecursiveRefinementConfig(
        budget=RefinementBudget(max_depth=5, max_width=4, max_total_updates=24, seed=2)
    )
    first = RecursiveRefinementController(first_config).run(_problem(), target_width=4)
    second = RecursiveRefinementController(second_config).run(_problem(), target_width=4)
    assert first.as_dict() != second.as_dict()
    assert first.trajectories and second.trajectories


def test_controller_respects_depth_width_and_update_budgets():
    config = RecursiveRefinementConfig(
        budget=RefinementBudget(max_depth=3, max_width=2, max_total_updates=8, seed=5)
    )
    run = RecursiveRefinementController(config).run(_problem(), target_width=4)
    assert run.max_observed_width <= 2
    assert run.max_observed_depth <= 3
    assert run.total_updates <= 8


def test_adaptive_halting_terminates_easy_cases_early():
    config = RecursiveRefinementConfig(
        budget=RefinementBudget(
            max_depth=8,
            max_width=1,
            max_total_updates=24,
            convergence_threshold=0.65,
            uncertainty_threshold=0.35,
            seed=0,
        )
    )
    run = RecursiveRefinementController(config).run(_problem(), target_width=1)
    assert run.max_observed_depth < 8
    assert any(trajectory.states[-1].halted for trajectory in run.trajectories)


def test_progressive_widening_can_be_disabled():
    config = RecursiveRefinementConfig(
        budget=RefinementBudget(max_depth=4, max_width=4, max_total_updates=24),
        progressive_widening=False,
    )
    run = RecursiveRefinementController(config).run(_problem(), target_width=4)
    assert len(run.trajectories) == 1
    assert run.max_observed_width == 1


def test_view_routing_can_be_disabled():
    config = RecursiveRefinementConfig(
        budget=RefinementBudget(max_depth=3, max_width=1, max_total_updates=8),
        routing=ViewRoutingConfig(enabled=False),
        adaptive_halting=False,
    )
    run = RecursiveRefinementController(config).run(_problem(), target_width=1)
    route = [op.view_name for op in run.trajectories[0].states[-1].operations]
    assert route
    assert set(route) == {"verification"}


def test_weak_branches_are_pruned_and_shared_prefix_is_accounted_once():
    config = RecursiveRefinementConfig(
        budget=RefinementBudget(
            max_depth=4,
            max_width=4,
            max_total_updates=24,
            prune_below=0.2,
            seed=11,
        )
    )
    run = RecursiveRefinementController(config).run(_problem(), target_width=4)
    assert any(trajectory.pruned for trajectory in run.trajectories)
    assert sum(len(trajectory.states) for trajectory in run.trajectories) > run.total_updates
