"""Self-play tests for semantic constraint logging."""

from __future__ import annotations

import json
from pathlib import Path

from rg_tracer.cli import main


def test_self_play_semantic_constraints_write_logs(tmp_path: Path, capsys, monkeypatch):
    problem = (
        Path(__file__).resolve().parents[1]
        / "datasets"
        / "semantic_constraints"
        / "finite_domain_requirements.jsonl"
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
            "lattice_trm",
            "--lattice-mode",
            "gated",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    run_dir = Path(output["run_dir"])
    assert (run_dir / "semantic_constraints.jsonl").exists()
    with (run_dir / "best.json").open("r", encoding="utf8") as handle:
        best = json.load(handle)
    assert best["semantic_constraint_overlay"]["safe_for_gated_projection"]
