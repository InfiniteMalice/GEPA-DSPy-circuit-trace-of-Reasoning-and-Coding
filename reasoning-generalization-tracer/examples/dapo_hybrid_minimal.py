"""Minimal CPU-safe DAPO hybrid training example with mocked components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _install_gepa_stubs() -> None:
    """Install shared GEPA stubs for local examples."""

    from rg_tracer.testing.gepa_stubs import install_gepa_stubs

    install_gepa_stubs()


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
        logger=JSONLLogger(Path(__file__).parent / "dapo_hybrid_minimal.jsonl"),
    )
    trainer.run()


if __name__ == "__main__":
    main()
