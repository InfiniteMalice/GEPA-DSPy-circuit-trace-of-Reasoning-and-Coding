"""Tests for recursive-refinement configuration validation."""

from __future__ import annotations

import json

import pytest

from rg_tracer.recursive_refinement import (
    LatticeConfig,
    RecursiveRefinementConfig,
    RefinementBudget,
    ViewRoutingConfig,
)
from rg_tracer.recursive_refinement.scoring import score_trajectory
from rg_tracer.recursive_refinement.types import RefinementState


def test_default_refinement_config_is_valid():
    config = RecursiveRefinementConfig()
    assert config.budget.max_depth == 8
    assert config.routing.enabled
    assert config.process_reward_weight == 0.0


def test_refinement_budget_rejects_invalid_values():
    with pytest.raises(ValueError, match="max_width"):
        RefinementBudget(max_width=0)
    with pytest.raises(ValueError, match="min_depth"):
        RefinementBudget(max_depth=2, min_depth=3)
    with pytest.raises(ValueError, match="uncertainty_threshold"):
        RefinementBudget(uncertainty_threshold=1.5)


def test_view_routing_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="route_strategy"):
        ViewRoutingConfig(route_strategy="diffusion_geometry")


def test_lattice_config_rejects_unknown_abstraction_source():
    with pytest.raises(ValueError, match="abstraction_source"):
        LatticeConfig(abstraction_source="typo")


def test_process_score_decomposition_is_json_serializable():
    state = RefinementState(
        trajectory_id="t",
        parent_id=None,
        depth=0,
        state_value=0.2,
        uncertainty=0.4,
        confidence=0.6,
        prediction=5,
        active_views=[],
        operations=[],
    )
    score = score_trajectory(
        [state],
        total_updates=0,
        max_total_updates=4,
        converged=False,
        budget_exhausted=False,
    )
    assert json.loads(json.dumps(score.as_dict()))["total"] == score.total
