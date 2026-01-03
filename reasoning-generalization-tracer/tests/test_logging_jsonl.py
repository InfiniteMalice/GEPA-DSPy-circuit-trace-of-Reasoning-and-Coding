"""Tests for JSONL logging schema."""

from __future__ import annotations

import json
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


def _install_gepa_stubs() -> None:
    gepa_module = types.ModuleType("gepa_dapo_grn")
    interfaces_module = types.ModuleType("gepa_dapo_grn.gepa_interfaces")
    policy_module = types.ModuleType("gepa_dapo_grn.policy_interfaces")

    @dataclass(frozen=True)
    class GEPAFeedback:
        rewards: Dict[str, float]
        tags: Dict[str, float]
        meta: Dict[str, str]
        abstained: bool

    interfaces_module.GEPAFeedback = GEPAFeedback

    class Policy:
        pass

    policy_module.Policy = Policy

    @dataclass(frozen=True)
    class DAPOConfig:
        learning_rate: float = 1e-4
        clip_ratio: float = 0.2
        kl_target: float = 0.1
        kl_coef: float = 0.1

    @dataclass(frozen=True)
    class GRNConfig:
        enable_policy: bool = False
        enable_value: bool = False
        enable_probes: bool = False

    @dataclass(frozen=True)
    class RewardMixerConfig:
        weights: Dict[str, float]

    class CurriculumTracker:
        def sample_weight(self, task_id: str | None) -> float:
            return 1.0

    class SafetyController:
        def __init__(self) -> None:
            self.latest = {}

    class DAPOTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def train_step(self, dapo_batch: object, feedbacks: object) -> Dict[str, float]:
            return {"loss": 0.1}

    gepa_module.DAPOConfig = DAPOConfig
    gepa_module.GRNConfig = GRNConfig
    gepa_module.RewardMixerConfig = RewardMixerConfig
    gepa_module.CurriculumTracker = CurriculumTracker
    gepa_module.SafetyController = SafetyController
    gepa_module.DAPOTrainer = DAPOTrainer
    gepa_module.gepa_interfaces = interfaces_module
    gepa_module.policy_interfaces = policy_module

    sys.modules["gepa_dapo_grn"] = gepa_module
    sys.modules["gepa_dapo_grn.gepa_interfaces"] = interfaces_module
    sys.modules["gepa_dapo_grn.policy_interfaces"] = policy_module


_install_gepa_stubs()

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
