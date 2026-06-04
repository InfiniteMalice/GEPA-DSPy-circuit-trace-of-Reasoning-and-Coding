"""Smoke test for recursive ladder experiment script output files."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.run_recursive_ladder import main


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
