"""Tests for the LRD grokking script argument handling."""

from __future__ import annotations

import pytest

from scripts.run_lrd_grokking_matrix import main


def test_lrd_grokking_script_rejects_negative_limit(capsys):
    with pytest.raises(SystemExit):
        main(["--limit", "-1"])
    assert "--limit must be non-negative" in capsys.readouterr().err
