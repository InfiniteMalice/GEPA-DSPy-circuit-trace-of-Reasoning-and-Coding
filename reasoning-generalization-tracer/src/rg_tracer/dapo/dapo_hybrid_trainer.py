"""Hybrid DAPO training loop integrating GEPA feedback with rg-tracer scoring."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from gepa_dapo_grn import (
    CurriculumTracker,
    DAPOConfig,
    DAPOTrainer,
    GRNConfig,
    RewardMixerConfig,
    SafetyController,
)
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback

from .feedback_adapter import FeedbackMappingConfig, make_gepa_feedback
from .logging import JSONLLogger, build_log_record

_POLICY_METHOD_ERROR = "Policy must implement generate_with_logprobs"


@dataclass(frozen=True)
class HybridTrainingConfig:
    dapo: DAPOConfig
    grn: GRNConfig
    reward_mixer: RewardMixerConfig
    group_size: int = 1
    eval_every: int = 100
    max_steps: Optional[int] = None
    seed: int = 0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError("HybridTrainingConfig.group_size must be > 0")
        if self.eval_every <= 0:
            raise ValueError("HybridTrainingConfig.eval_every must be > 0")


class DAPOHybridTrainer:
    def __init__(
        self,
        *,
        policy: Any,
        scorer: Any,
        dataloader: Iterable[Mapping[str, Any]],
        feedback_cfg: FeedbackMappingConfig,
        cfg: HybridTrainingConfig,
        logger: Optional[JSONLLogger] = None,
        trainer: Optional[DAPOTrainer] = None,
        curriculum: Optional[CurriculumTracker] = None,
        safety: Optional[SafetyController] = None,
    ) -> None:
        self.policy = policy
        self.scorer = scorer
        self.dataloader = dataloader
        self.feedback_cfg = feedback_cfg
        self.cfg = cfg
        self.curriculum = curriculum or CurriculumTracker()
        self.safety = safety or SafetyController()
        self.trainer = trainer or DAPOTrainer(
            policy=policy,
            dapo_config=cfg.dapo,
            reward_mixer_config=cfg.reward_mixer,
            grn_config=cfg.grn,
            curriculum_tracker=self.curriculum,
            safety_controller=self.safety,
        )
        self.logger = logger

    def run(self, *, eval_fn: Optional[Any] = None) -> None:
        """Run the training loop.

        Note: if the dataloader is a generator, it can only be iterated once.
        """
        for step, batch in enumerate(self.dataloader):
            if self.cfg.max_steps is not None and step >= self.cfg.max_steps:
                break
            prompts = list(batch.get("prompts", []))
            if not prompts:
                continue
            task_ids = list(batch.get("task_ids", [None] * len(prompts)))
            prompt_ids = list(batch.get("prompt_ids", [None] * len(prompts)))
            weights = [self.curriculum.sample_weight(task_id) for task_id in task_ids]

            completions, actions, old_logprobs, gen_meta = self._generate(prompts)
            prompt_repeated, task_repeated, prompt_id_repeated = _repeat_by_group(
                prompts,
                task_ids,
                prompt_ids,
                self.cfg.group_size,
            )

            local_metrics_list = self.scorer.score(prompt_repeated, completions)
            feedbacks = self._make_feedbacks(
                prompt_repeated,
                completions,
                local_metrics_list,
                task_repeated,
                prompt_id_repeated,
            )

            expanded_weights = _repeat(weights, self.cfg.group_size)
            dapo_batch = {
                "obs": prompt_repeated,
                "actions": actions,
                "old_logprobs": old_logprobs,
                "sample_weights": expanded_weights,
            }
            logs = self.trainer.train_step(dapo_batch, feedbacks)
            if self.logger is not None:
                record = build_log_record(
                    step=step,
                    rl_metrics=logs,
                    feedbacks=feedbacks,
                    generation_metadata=gen_meta,
                    curriculum={
                        "task_id": task_repeated,
                        "sample_weight": expanded_weights,
                    },
                    safety=getattr(self.safety, "latest", None) or {},
                )
                self.logger.write(record)

            if eval_fn and step % self.cfg.eval_every == 0:
                eval_fn(step, self.policy)

    def _generate(
        self, prompts: Sequence[str]
    ) -> Tuple[List[str], List[List[int]], List[float], List[Dict[str, Any]]]:
        gen = getattr(self.policy, "generate_with_logprobs", None)
        if not callable(gen):
            raise TypeError(_POLICY_METHOD_ERROR)

        output = gen(
            prompts,
            group_size=self.cfg.group_size,
            temperature=self.cfg.temperature,
            seed=self.cfg.seed,
        )
        completions = list(_get_attr(output, "completions"))
        actions = list(_get_attr(output, "actions"))
        old_logprobs = list(_get_attr(output, "logprobs"))
        metadata = list(_get_attr(output, "metadata"))

        prompt_count = len(prompts)
        expected_len = prompt_count * self.cfg.group_size
        _validate_generation_lengths(
            expected_len,
            prompt_count=prompt_count,
            group_size=self.cfg.group_size,
            completions=completions,
            actions=actions,
            logprobs=old_logprobs,
            metadata=metadata,
        )

        generation_metadata = []
        repeated_prompts = _repeat(prompts, self.cfg.group_size)
        for prompt_index, (prompt, completion, meta) in enumerate(
            zip(
                repeated_prompts,
                completions,
                metadata,
                strict=True,
            )
        ):
            if not isinstance(meta, Mapping):
                raise TypeError(
                    "Generation metadata must be a mapping at index "
                    f"{prompt_index}: {meta!r}"
                )
            _validate_prompt_grouping(
                prompt_index,
                prompt,
                meta,
                self.cfg.group_size,
            )
            prompt_hash = _hash_text(prompt)
            completion_hash = _hash_text(completion)
            generation_metadata.append(
                {
                    "prompt_id": prompt_hash,
                    "completion_id": completion_hash,
                    "length": len(completion),
                    "seed": meta.get("seed"),
                    "temperature": meta.get("temperature"),
                }
            )
        return completions, actions, old_logprobs, generation_metadata

    def _make_feedbacks(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        local_metrics_list: Sequence[Mapping[str, float]],
        task_ids: Sequence[Optional[str]],
        prompt_ids: Sequence[Optional[str]],
    ) -> List[GEPAFeedback]:
        feedbacks: List[GEPAFeedback] = []
        for prompt, completion, metrics, task_id, prompt_id in zip(
            prompts,
            completions,
            local_metrics_list,
            task_ids,
            prompt_ids,
            strict=True,
        ):
            meta = {}
            if task_id is not None:
                meta[self.feedback_cfg.task_id_field] = task_id
            if prompt_id is not None:
                meta[self.feedback_cfg.prompt_id_field] = prompt_id
            feedbacks.append(
                make_gepa_feedback(
                    prompt=prompt,
                    completion=completion,
                    local_metrics=dict(metrics),
                    meta=meta,
                    cfg=self.feedback_cfg,
                )
            )
        return feedbacks


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _repeat(items: Sequence[Any], group_size: int) -> List[Any]:
    repeated: List[Any] = []
    for item in items:
        repeated.extend([item] * group_size)
    return repeated


def _repeat_by_group(
    prompts: Sequence[str],
    task_ids: Sequence[Optional[str]],
    prompt_ids: Sequence[Optional[str]],
    group_size: int,
) -> Tuple[List[str], List[Optional[str]], List[Optional[str]]]:
    return (
        _repeat(prompts, group_size),
        _repeat(list(task_ids), group_size),
        _repeat(list(prompt_ids), group_size),
    )


def _get_attr(output: Any, name: str) -> Sequence[Any]:
    if not hasattr(output, name):
        raise ValueError(f"Generation output missing attribute: {name}")
    return getattr(output, name)


def _validate_generation_lengths(
    expected_len: int,
    *,
    prompt_count: int,
    group_size: int,
    **fields: Sequence[Any],
) -> None:
    for name, values in fields.items():
        if len(values) != expected_len:
            raise ValueError(
                f"Generation output {name} length {len(values)} does not match "
                f"expected length {expected_len} "
                f"(prompt_count={prompt_count}, group_size={group_size})."
            )


def _validate_prompt_grouping(
    index: int,
    prompt: str,
    meta: Mapping[str, Any],
    group_size: int,
) -> None:
    expected_prompt_index = index // group_size
    if "prompt_index" in meta:
        if int(meta["prompt_index"]) != expected_prompt_index:
            raise ValueError(
                "Generation metadata prompt_index does not match expected grouping: "
                f"{meta['prompt_index']} != {expected_prompt_index}"
            )
        return
    if "prompt" in meta:
        if meta["prompt"] != prompt:
            raise ValueError(
                "Generation metadata prompt does not match expected prompt"
            )
        return
    raise ValueError(
        "Generation metadata must include prompt_index or prompt to validate grouping"
    )
