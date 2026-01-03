"""Smoke test for DAPOHybridTrainer train_step integration."""

from __future__ import annotations

import json
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


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
        def sample_weight(self, task_id: Optional[str]) -> float:
            return 1.0

    class SafetyController:
        def __init__(self) -> None:
            self.latest = {"ema_deception": 0.0}

    class DAPOTrainer:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def train_step(
            self, dapo_batch: Mapping[str, Any], feedbacks: Iterable[Any]
        ) -> Dict[str, Any]:
            return {
                "loss": 0.1,
                "policy_loss": 0.05,
                "kl": 0.01,
                "clip_ratio": 0.2,
                "lr": 1e-4,
                "grad_norm": 0.5,
            }

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

from rg_tracer.dapo import DAPOHybridTrainer, FeedbackMappingConfig, HybridTrainingConfig
from rg_tracer.dapo.logging import JSONLLogger
from gepa_dapo_grn import DAPOConfig, GRNConfig, RewardMixerConfig


@dataclass
class DummyGeneration:
    completions: list
    actions: list
    logprobs: list
    metadata: list


class DummyPolicy:
    def generate_with_logprobs(
        self,
        prompts: list,
        *,
        group_size: int = 1,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> DummyGeneration:
        return DummyGeneration(
            completions=["ok" for _ in prompts],
            actions=[[0] for _ in prompts],
            logprobs=[0.0 for _ in prompts],
            metadata=[
                {
                    "prompt": prompt,
                    "completion": "ok",
                    "temperature": temperature,
                    "seed": seed,
                }
                for prompt in prompts
            ],
        )


class DummyScorer:
    def score(self, prompts: Iterable[str], completions: Iterable[str]) -> list:
        return [{"reward": 1.0} for _ in zip(prompts, completions)]


def test_train_step_smoke() -> None:
    dataloader = [{"prompts": ["Hello"], "task_ids": ["t1"], "prompt_ids": ["p1"]}]
    feedback_cfg = FeedbackMappingConfig(reward_keys={"reward": "reward"}, tag_keys={})

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = Path(tmp_dir) / "train.jsonl"
        trainer = DAPOHybridTrainer(
            policy=DummyPolicy(),
            scorer=DummyScorer(),
            dataloader=dataloader,
            feedback_cfg=feedback_cfg,
            cfg=HybridTrainingConfig(
                dapo=DAPOConfig(),
                grn=GRNConfig(),
                reward_mixer=RewardMixerConfig(weights={"reward": 1.0}),
                max_steps=1,
            ),
            logger=JSONLLogger(log_path),
        )
        trainer.run()

        logged = json.loads(log_path.read_text().splitlines()[0])
        assert logged["rl"]["loss"] > 0.0
        assert logged["gepa"][0]["rewards"]["reward"] == 1.0
