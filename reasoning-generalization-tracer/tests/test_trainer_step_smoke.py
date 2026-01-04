"""Smoke test for DAPOHybridTrainer train_step integration."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from tests.gepa_test_stubs import install_gepa_stubs

install_gepa_stubs()

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
