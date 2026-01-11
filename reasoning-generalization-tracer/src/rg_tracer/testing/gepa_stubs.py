"""Shared GEPA stubs for local tests and examples."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Dict, Optional

_GEPA_MODULE_NAME = "gepa_dapo_grn"
_GEPA_INTERFACES_NAME = "gepa_dapo_grn.gepa_interfaces"
_GEPA_POLICY_NAME = "gepa_dapo_grn.policy_interfaces"


def _ensure_module(name: str) -> types.ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _ensure_attr(module: types.ModuleType, name: str, value: object) -> None:
    if not hasattr(module, name):
        setattr(module, name, value)


def install_gepa_stubs() -> None:
    gepa_module = _ensure_module(_GEPA_MODULE_NAME)
    if not hasattr(gepa_module, "__path__"):
        gepa_module.__path__ = []
    interfaces_module = _ensure_module(_GEPA_INTERFACES_NAME)
    policy_module = _ensure_module(_GEPA_POLICY_NAME)

    @dataclass(frozen=True)
    class GEPAFeedback:
        rewards: Dict[str, float]
        tags: Dict[str, float]
        meta: Dict[str, str]
        abstained: bool

    _ensure_attr(interfaces_module, "GEPAFeedback", GEPAFeedback)

    class Policy:
        pass

    _ensure_attr(policy_module, "Policy", Policy)

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
        def sample_weight(self, _task_id: Optional[str]) -> float:
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

        def train_step(
            self, _dapo_batch: object, _feedbacks: object
        ) -> Dict[str, float]:
            return {"loss": 0.1}

    _ensure_attr(gepa_module, "DAPOConfig", DAPOConfig)
    _ensure_attr(gepa_module, "GRNConfig", GRNConfig)
    _ensure_attr(gepa_module, "RewardMixerConfig", RewardMixerConfig)
    _ensure_attr(gepa_module, "CurriculumTracker", CurriculumTracker)
    _ensure_attr(gepa_module, "SafetyController", SafetyController)
    _ensure_attr(gepa_module, "DAPOTrainer", DAPOTrainer)
    _ensure_attr(gepa_module, "gepa_interfaces", interfaces_module)
    _ensure_attr(gepa_module, "policy_interfaces", policy_module)
