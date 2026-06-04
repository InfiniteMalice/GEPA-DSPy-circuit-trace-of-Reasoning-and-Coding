"""CLI tests for recursive ladder flags."""

from __future__ import annotations

import json
from pathlib import Path

from rg_tracer.cli import main


def test_cli_accepts_lattice_ptrm_flags(tmp_path, monkeypatch, capsys):
    problem = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "constraint_lattice"
        / "finite_domain.jsonl"
    )
    monkeypatch.chdir(tmp_path)
    main(
        [
            "self-play",
            "--profile",
            "proof_math",
            "--problem",
            str(problem),
            "--sampler",
            "lattice_ptrm",
            "--lattice-mode",
            "gated",
            "--trajectory-count",
            "3",
            "--noise-std",
            "0.10",
            "--seed",
            "7",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    run_dir = Path(output["run_dir"])
    assert (run_dir / "lattice_diagnostics.jsonl").exists()
    assert (run_dir / "perturbations.jsonl").exists()
