"""Smoke test for recursive ladder experiment script output files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.run_recursive_ladder import _summarize, main


def test_recursive_ladder_script_writes_summaries(tmp_path, monkeypatch):
    dataset = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "constraint_lattice"
        / "finite_domain.jsonl"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_recursive_ladder.py",
            "--dataset",
            str(dataset),
            "--seeds",
            "0",
            "--output-root",
            str(tmp_path),
        ],
    )
    main()
    output_dirs = list(tmp_path.iterdir())
    assert output_dirs
    out = output_dirs[0]
    assert (out / "results.jsonl").exists()
    assert (out / "summary.csv").exists()
    assert (out / "summary.md").exists()


def test_recursive_ladder_summary_counts_correctness_and_representative_branch_totals(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {
            "predicted_answer": 0,
            "ground_truth": 0,
            "abstained": False,
            "trajectory_id": "a",
            "total_updates": 1,
            "max_depth": 1,
            "max_width": 2,
            "lattice_diagnostics": {
                "projection_count": 1,
                "merged_branch_count": 3,
                "pruned_branch_count": 2,
            },
        },
        {
            "predicted_answer": 1,
            "ground_truth": 0,
            "abstained": False,
            "trajectory_id": "b",
            "total_updates": 1,
            "max_depth": 1,
            "max_width": 2,
            "lattice_diagnostics": {
                "projection_count": 1,
                "merged_branch_count": 3,
                "pruned_branch_count": 2,
            },
        },
        {
            "predicted_answer": None,
            "ground_truth": 0,
            "abstained": True,
            "trajectory_id": None,
            "lattice_diagnostics": {"is_summary": True, "merged_branch_count": 99},
        },
    ]
    (run_dir / "scores.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf8",
    )
    summary = _summarize(
        sampler="lattice_ptrm",
        seed=0,
        dataset="dummy",
        run_dir=run_dir,
        wall_clock_seconds=0.0,
    )
    assert summary["accuracy"] == 1 / 3
    assert summary["merged_branch_count"] == 3
    assert summary["pruned_branch_count"] == 2
