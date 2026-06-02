"""Self-play integration tests for the experimental gram_mdt sampler."""

from __future__ import annotations

import json
from pathlib import Path

from rg_tracer.concepts import ConceptSpec
from rg_tracer.recursive_refinement import RecursiveRefinementConfig, RefinementBudget
from rg_tracer.runners.self_play import run_self_play


def _problem_path() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets" / "toy_math" / "addition_small.jsonl"


def test_self_play_gram_mdt_creates_recursive_artifacts(tmp_path):
    config = RecursiveRefinementConfig(
        budget=RefinementBudget(max_depth=4, max_width=4, max_total_updates=24, seed=7)
    )
    result = run_self_play(
        _problem_path(),
        profile="proof_math",
        k=4,
        sampler="gram_mdt",
        concept=ConceptSpec(name="parity", definition="Parity", expected_substructures=["parity"]),
        output_dir=tmp_path,
        refinement_config=config,
    )
    run_dir = Path(result["run_dir"])
    assert (run_dir / "trajectories.jsonl").exists()
    assert (run_dir / "view_routes.jsonl").exists()
    assert (run_dir / "budget_metrics.json").exists()
    assert "Recursive Refinement Metrics" in (run_dir / "summary.md").read_text()
    budget = json.loads((run_dir / "budget_metrics.json").read_text())
    assert budget["max_observed_width"] <= 4
    scores = [
        json.loads(line)
        for line in (run_dir / "scores.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert all("grn_flags" in score for score in scores)
    assert all("trajectory_id" in score for score in scores)


def test_default_process_reward_weight_does_not_change_trm_ranking(tmp_path):
    first = run_self_play(
        _problem_path(),
        profile="proof_math",
        k=2,
        sampler="trm",
        output_dir=tmp_path / "a",
    )
    second = run_self_play(
        _problem_path(),
        profile="proof_math",
        k=2,
        sampler="trm",
        output_dir=tmp_path / "b",
        process_reward_weight=1.0,
    )
    assert first["best"].predicted_answer == second["best"].predicted_answer
    assert first["best"].composite == second["best"].composite


def test_process_bonus_does_not_override_hard_gates_or_abstention(tmp_path):
    config = RecursiveRefinementConfig(
        budget=RefinementBudget(max_depth=2, max_width=2, max_total_updates=6, seed=1)
    )
    result = run_self_play(
        _problem_path(),
        profile="proof_math",
        k=2,
        sampler="gram_mdt",
        output_dir=tmp_path,
        refinement_config=config,
        process_reward_weight=1.0,
    )
    for candidate in result["candidates"]:
        if not candidate.passes_gates or candidate.abstained:
            assert candidate.process_bonus == 0.0
