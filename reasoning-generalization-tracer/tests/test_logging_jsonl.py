"""Tests for JSONL logging schema."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict

from tests.gepa_test_stubs import install_gepa_stubs


install_gepa_stubs()

from rg_tracer.dapo.logging import JSONLLogger, build_log_record
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback


def test_logging_jsonl_schema() -> None:
    feedback = GEPAFeedback(
        rewards={"correctness": 1.0},
        tags={"format_penalty": 0.1},
        meta={"task_id": "t1", "prompt_id": "p1"},
        abstained=False,
    )
    record = build_log_record(
        step=1,
        rl_metrics={"loss": 0.1, "policy_loss": 0.05},
        feedbacks=[feedback],
        generation_metadata=[{"prompt_id": "p1", "completion_id": "c1", "length": 4}],
        curriculum={"task_id": ["t1"], "sample_weight": [1.0]},
        safety={"ema_deception": 0.0},
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "log.jsonl"
        logger = JSONLLogger(path)
        logger.write(record)

        loaded = json.loads(path.read_text().splitlines()[0])
        assert "rl" in loaded
        assert "curriculum" in loaded
        assert "safety" in loaded
        assert "gepa" in loaded
        assert loaded["gepa"][0]["rewards"]["correctness"] == 1.0
        assert "generation" in loaded
