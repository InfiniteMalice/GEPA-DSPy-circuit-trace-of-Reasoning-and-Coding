"""Minimal CPU-safe DAPO hybrid training example with mocked components."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


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
            self.latest = {
                "ema_deception": 0.0,
                "ema_harm": 0.0,
                "ema_calibration_err": 0.0,
            }

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


@dataclass
class DummyGeneration:
    completions: List[str]
    actions: List[List[int]]
    logprobs: List[float]
    metadata: List[Dict[str, Any]]


class DummyPolicy:
    def generate_with_logprobs(
        self,
        prompts: List[str],
        *,
        group_size: int = 1,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> DummyGeneration:
        completions: List[str] = []
        actions: List[List[int]] = []
        logprobs: List[float] = []
        metadata: List[Dict[str, Any]] = []
        for prompt in prompts:
            for index in range(group_size):
                completion = f"answer-{index}"
                completions.append(completion)
                actions.append([index])
                logprobs.append(-0.1 * index)
                metadata.append(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "temperature": temperature,
                        "seed": seed,
                    }
                )
        return DummyGeneration(
            completions=completions,
            actions=actions,
            logprobs=logprobs,
            metadata=metadata,
        )


class DummyScorer:
    def score(
        self, prompts: Iterable[str], completions: Iterable[str]
    ) -> List[Dict[str, float]]:
        scores = []
        for _, completion in zip(prompts, completions, strict=True):
            scores.append({"correctness": 1.0, "length": float(len(completion))})
        return scores


def main() -> None:
    _install_gepa_stubs()

    from gepa_dapo_grn import DAPOConfig, GRNConfig, RewardMixerConfig
    from rg_tracer.dapo import (
        DAPOHybridTrainer,
        FeedbackMappingConfig,
        HybridTrainingConfig,
    )
    from rg_tracer.dapo.logging import JSONLLogger

    dataloader = [
        {"prompts": ["Add 1 + 1"], "task_ids": ["toy"], "prompt_ids": ["p1"]},
        {"prompts": ["Add 2 + 2"], "task_ids": ["toy"], "prompt_ids": ["p2"]},
    ]
    feedback_cfg = FeedbackMappingConfig(
        reward_keys={"correctness": "correctness"},
        tag_keys={"length": "length"},
    )
    trainer = DAPOHybridTrainer(
        policy=DummyPolicy(),
        scorer=DummyScorer(),
        dataloader=dataloader,
        feedback_cfg=feedback_cfg,
        cfg=HybridTrainingConfig(
            dapo=DAPOConfig(),
            grn=GRNConfig(),
            reward_mixer=RewardMixerConfig(weights={"correctness": 1.0}),
            group_size=2,
            max_steps=2,
        ),
        logger=JSONLLogger(Path("examples/dapo_hybrid_minimal.jsonl")),
    )
    trainer.run()


if __name__ == "__main__":
    main()
