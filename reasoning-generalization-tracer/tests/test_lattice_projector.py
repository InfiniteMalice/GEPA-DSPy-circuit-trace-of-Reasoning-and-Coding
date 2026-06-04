"""Tests for LDT-inspired explicit lattice projection."""

from __future__ import annotations

from rg_tracer.recursive_refinement import LatticeConfig, LatticeDeductionProjector


def test_projector_detects_unique_resolution():
    projector = LatticeDeductionProjector(LatticeConfig(mode="gated"))
    result = projector.project(
        {
            "task": "finite_domain_constraint",
            "domain": [1, 2, 3, 4],
            "constraints": [
                {"id": "even", "allowed": [2, 4]},
                {"id": "gt2", "allowed": [3, 4]},
            ],
        }
    )
    assert result.resolved
    assert result.remaining_candidates == [4]
    assert not result.abstain_recommended


def test_projector_abstains_on_contradiction_in_gated_mode():
    projector = LatticeDeductionProjector(LatticeConfig(mode="gated"))
    result = projector.project(
        {
            "task": "finite_domain_constraint",
            "domain": [1, 2, 3, 4],
            "constraints": [
                {"id": "even", "allowed": [2, 4]},
                {"id": "odd", "allowed": [1, 3]},
            ],
        }
    )
    assert result.contradiction_detected
    assert result.abstain_recommended
    assert result.abstain_reason == "contradiction"


def test_shadow_mode_records_unresolved_without_gating():
    projector = LatticeDeductionProjector(LatticeConfig(mode="shadow"))
    result = projector.project(
        {
            "task": "finite_domain_constraint",
            "domain": [1, 2, 3, 4],
            "constraints": [{"id": "even", "allowed": [2, 4]}],
        }
    )
    assert result.unresolved
    assert result.abstain_recommended
