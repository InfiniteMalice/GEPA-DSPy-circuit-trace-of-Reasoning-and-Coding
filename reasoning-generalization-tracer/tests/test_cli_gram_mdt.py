"""CLI tests for gram_mdt self-play flags."""

from __future__ import annotations

import json
from pathlib import Path

from rg_tracer.cli import main


def test_cli_accepts_gram_mdt_flags(tmp_path, monkeypatch, capsys):
    problem = Path(__file__).resolve().parents[1] / "datasets" / "toy_math" / "addition_small.jsonl"
    monkeypatch.chdir(tmp_path)
    main(
        [
            "self-play",
            "--profile",
            "proof_math",
            "--problem",
            str(problem),
            "--concept",
            "parity",
            "--sampler",
            "gram_mdt",
            "--k",
            "2",
            "--max-depth",
            "3",
            "--max-width",
            "2",
            "--max-total-updates",
            "8",
            "--branch-factor",
            "2",
            "--seed",
            "7",
            "--disable-view-routing",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert Path(output["run_dir"]).exists()
