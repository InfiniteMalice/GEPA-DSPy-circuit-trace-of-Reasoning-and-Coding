"""Smoke test for DAPOHybridTrainer train_step integration."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from gepa_dapo_grn import DAPOConfig, GRNConfig, RewardMixerConfig

from rg_tracer.dapo import (
    DAPOHybridTrainer,
    FeedbackMappingConfig,
    HybridTrainingConfig,
)
from rg_tracer.dapo.logging import JSONLLogger


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
        completions = []
        actions = []
        logprobs = []
        metadata = []
        for prompt in prompts:
            for _ in range(group_size):
                completions.append("ok")
                actions.append([0])
                logprobs.append(0.0)
                metadata.append(
                    {
                        "prompt": prompt,
                        "completion": "ok",
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
    def score(self, prompts: Iterable[str], completions: Iterable[str]) -> list:
        return [{"reward": 1.0} for _ in zip(prompts, completions, strict=True)]


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
