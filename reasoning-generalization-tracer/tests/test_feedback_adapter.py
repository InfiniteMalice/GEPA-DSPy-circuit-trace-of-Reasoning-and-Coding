"""Tests for GEPA feedback mapping adapter."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass


def _install_gepa_stubs() -> None:
    gepa_module = types.ModuleType("gepa_dapo_grn")
    interfaces_module = types.ModuleType("gepa_dapo_grn.gepa_interfaces")
    policy_module = types.ModuleType("gepa_dapo_grn.policy_interfaces")

    @dataclass(frozen=True)
    class GEPAFeedback:
        rewards: dict
        tags: dict
        meta: dict
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
        weights: dict

    class CurriculumTracker:
        def sample_weight(self, task_id: str | None) -> float:
            return 1.0

    class SafetyController:
        def __init__(self) -> None:
            self.latest = {}

    class DAPOTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def train_step(self, dapo_batch: object, feedbacks: object) -> dict:
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

from rg_tracer.dapo.feedback_adapter import FeedbackMappingConfig, make_gepa_feedback


def test_make_gepa_feedback_mapping() -> None:
    cfg = FeedbackMappingConfig(
        reward_keys={"correct": "correctness"},
        tag_keys={"fmt": "format_penalty"},
        abstain_field="abstain",
    )
    feedback = make_gepa_feedback(
        prompt="p",
        completion="c",
        local_metrics={"correct": 1.0, "fmt": 0.2, "abstain": 1.0},
        meta={"task_id": "t1", "prompt_id": "p1"},
        cfg=cfg,
    )

    assert feedback.rewards == {"correctness": 1.0}
    assert feedback.tags == {"format_penalty": 0.2}
    assert feedback.meta["task_id"] == "t1"
    assert feedback.meta["prompt_id"] == "p1"
    assert feedback.abstained is True
