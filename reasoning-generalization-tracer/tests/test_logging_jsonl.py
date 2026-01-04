"""Tests for JSONL logging schema."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from gepa_dapo_grn.gepa_interfaces import GEPAFeedback

from rg_tracer.dapo.logging import JSONLLogger, build_log_record


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
        assert loaded["step"] == 1
        assert "rl" in loaded
        assert loaded["rl"]["loss"] == 0.1
        assert loaded["rl"]["policy_loss"] == 0.05
        assert "curriculum" in loaded
        assert loaded["curriculum"]["task_id"] == ["t1"]
        assert "safety" in loaded
        assert loaded["safety"]["ema_deception"] == 0.0
        assert "gepa" in loaded
        assert loaded["gepa"][0]["rewards"]["correctness"] == 1.0
        assert loaded["gepa"][0]["tags"]["format_penalty"] == 0.1
        assert loaded["gepa"][0]["meta"]["task_id"] == "t1"
        assert loaded["gepa"][0]["abstained"] is False
        assert "generation" in loaded
        assert loaded["generation"][0]["prompt_id"] == "p1"
