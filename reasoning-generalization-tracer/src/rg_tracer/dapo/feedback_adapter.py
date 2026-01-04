"""Adapters for mapping local scoring metrics into GEPA feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from gepa_dapo_grn.gepa_interfaces import GEPAFeedback


@dataclass(frozen=True)
class FeedbackMappingConfig:
    reward_keys: Dict[str, str]
    tag_keys: Dict[str, str]
    task_id_field: str = "task_id"
    prompt_id_field: str = "prompt_id"
    abstain_field: Optional[str] = None


def _coerce_float(value: Any) -> float:
    return float(value)


def make_gepa_feedback(
    prompt: str,
    completion: str,
    local_metrics: Dict[str, float],
    meta: Dict[str, str],
    cfg: FeedbackMappingConfig,
) -> GEPAFeedback:
    rewards: Dict[str, float] = {}
    tags: Dict[str, float] = {}
    for local_key, gepa_key in cfg.reward_keys.items():
        if local_key in local_metrics:
            rewards[gepa_key] = _coerce_float(local_metrics[local_key])
    for local_key, gepa_key in cfg.tag_keys.items():
        if local_key in local_metrics:
            tags[gepa_key] = _coerce_float(local_metrics[local_key])

    meta_out = dict(meta)
    if cfg.task_id_field in meta_out and "task_id" not in meta_out:
        meta_out["task_id"] = str(meta_out[cfg.task_id_field])
    if cfg.prompt_id_field in meta_out and "prompt_id" not in meta_out:
        meta_out["prompt_id"] = str(meta_out[cfg.prompt_id_field])
    meta_out.setdefault("prompt", prompt)
    meta_out.setdefault("completion", completion)

    abstained = False
    if cfg.abstain_field:
        if cfg.abstain_field in local_metrics:
            abstained = bool(local_metrics[cfg.abstain_field])
        elif cfg.abstain_field in meta_out:
            abstained = bool(meta_out[cfg.abstain_field])

    return GEPAFeedback(
        rewards=rewards,
        tags=tags,
        meta=meta_out,
        abstained=abstained,
    )
