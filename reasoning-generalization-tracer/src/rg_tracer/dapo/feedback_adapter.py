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


_COERCE_ERROR_MSG = "Cannot coerce {type_name} to float: {value!r}"
_COLLISION_ERROR_MSG = "task_id_field and prompt_id_field must not collide"
_META_OVERWRITE_ERROR_MSG = "metadata contains both {custom} and {canonical} keys"
_FALSE_STRINGS = {"false", "0", "no", "none"}
_TRUE_STRINGS = {"true", "1", "yes"}


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            _COERCE_ERROR_MSG.format(
                type_name=type(value).__name__,
                value=value,
            )
        ) from exc


def _parse_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _FALSE_STRINGS:
            return False
        if lowered in _TRUE_STRINGS:
            return True
        return False
    return bool(value)


def make_gepa_feedback(
    prompt: str,
    completion: str,
    local_metrics: Dict[str, float],
    meta: Dict[str, str],
    cfg: FeedbackMappingConfig,
) -> GEPAFeedback:
    if cfg.task_id_field == cfg.prompt_id_field:
        raise ValueError(_COLLISION_ERROR_MSG)

    rewards: Dict[str, float] = {}
    tags: Dict[str, float] = {}
    for local_key, gepa_key in cfg.reward_keys.items():
        if local_key in local_metrics:
            rewards[gepa_key] = _coerce_float(local_metrics[local_key])
    for local_key, gepa_key in cfg.tag_keys.items():
        if local_key in local_metrics:
            tags[gepa_key] = _coerce_float(local_metrics[local_key])

    meta_out = dict(meta)
    meta_raw = dict(meta_out)
    # Normalize custom field names to canonical keys if different.
    if cfg.task_id_field != "task_id" and cfg.task_id_field in meta_out:
        if "task_id" in meta_out:
            raise ValueError(
                _META_OVERWRITE_ERROR_MSG.format(
                    custom=cfg.task_id_field,
                    canonical="task_id",
                )
            )
        meta_out["task_id"] = str(meta_out[cfg.task_id_field])
        del meta_out[cfg.task_id_field]
    if cfg.prompt_id_field != "prompt_id" and cfg.prompt_id_field in meta_out:
        if "prompt_id" in meta_out:
            raise ValueError(
                _META_OVERWRITE_ERROR_MSG.format(
                    custom=cfg.prompt_id_field,
                    canonical="prompt_id",
                )
            )
        meta_out["prompt_id"] = str(meta_out[cfg.prompt_id_field])
        del meta_out[cfg.prompt_id_field]
    meta_out = {key: str(value) for key, value in meta_out.items()}
    meta_out.setdefault("prompt", str(prompt))
    meta_out.setdefault("completion", str(completion))

    abstained = False
    if cfg.abstain_field:
        if cfg.abstain_field in local_metrics:
            abstained = _parse_boolish(local_metrics[cfg.abstain_field])
        elif cfg.abstain_field in meta_raw:
            abstained = _parse_boolish(meta_raw[cfg.abstain_field])
        elif cfg.abstain_field in meta_out:
            abstained = _parse_boolish(meta_out[cfg.abstain_field])

    return GEPAFeedback(
        rewards=rewards,
        tags=tags,
        meta=meta_out,
        abstained=abstained,
    )
