"""CLI tests for semantic constraint compilation."""

from __future__ import annotations

import json

from rg_tracer.cli import main


def test_compile_constraints_cli(capsys):
    main(
        [
            "compile-constraints",
            "--text",
            "The answer must be even and greater than 2.",
            "--domain",
            "1",
            "2",
            "3",
            "4",
            "--mode",
            "shadow",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert output["remaining_candidates"] == [4]
    assert not output["gated"]
