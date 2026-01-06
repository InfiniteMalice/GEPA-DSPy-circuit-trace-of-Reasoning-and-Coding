"""Shared GEPA stubs for local tests and examples."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Dict, Optional


def install_gepa_stubs() -> None:
    if "gepa_dapo_grn" in sys.modules:
        return

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
        def sample_weight(self, task_id: Optional[str]) -> float:
            return 1.0

    class SafetyController:
        def __init__(self) -> None:
            self.latest = {
                "ema_deception": 0.0,
                "ema_harm": 0.0,
                "ema_calibration_err": 0.0,
            }

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
