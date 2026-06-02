"""Tests for GRAMMDTSampler candidate compatibility."""

from __future__ import annotations

from rg_tracer.recursive_refinement import GRAMMDTSampler, RecursiveRefinementConfig
from rg_tracer.recursive_refinement import RefinementBudget


def test_gram_mdt_sampler_outputs_self_play_candidate_fields():
    config = RecursiveRefinementConfig(
        budget=RefinementBudget(max_depth=4, max_width=3, max_total_updates=16, seed=7)
    )
    candidates = GRAMMDTSampler(config).generate(
        {"id": "p", "task": "addition", "numbers": [2, 3], "answer": 5},
        k=3,
    )
    assert candidates
    assert {"text", "confidence", "metrics", "trace", "prediction"} <= set(candidates[0])
    assert "refinement_run" in candidates[0]
    assert "process_score_components" in candidates[0]
    assert candidates[0]["max_width"] <= 3
