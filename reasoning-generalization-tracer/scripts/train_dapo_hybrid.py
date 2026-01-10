"""CLI entrypoint for hybrid DAPO training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml
from gepa_dapo_grn import (
    CurriculumTracker,
    DAPOConfig,
    GRNConfig,
    RewardMixerConfig,
    SafetyController,
)

from rg_tracer.dapo import (
    DAPOHybridTrainer,
    FeedbackMappingConfig,
    HFPolicyAdapter,
    HybridTrainingConfig,
    JSONLLogger,
)
from rg_tracer.modules.torch_stub import torch

LOGGER = logging.getLogger(__name__)
_UNSUPPORTED_DTYPE_ERROR = "Unsupported torch dtype"


class NullScorer:
    def score(
        self, prompts: Iterable[str], completions: Iterable[str]
    ) -> List[Dict[str, float]]:
        LOGGER.warning("NullScorer in use; rewards will be empty")
        return [{} for _ in zip(prompts, completions, strict=True)]


def _load_jsonl(path: Path) -> List[Mapping[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {line}"
                ) from exc
    return rows


def _extract_prompt(record: Mapping[str, Any]) -> str:
    for key in ("prompt", "question", "problem", "text"):
        if key in record:
            return str(record[key])
    LOGGER.warning("No standard prompt key found in record; using JSON serialization")
    return json.dumps(record)


def _batch_records(
    records: List[Mapping[str, Any]], batch_size: int
) -> Iterable[Dict[str, Any]]:
    """Yield batches of records.

    Note: returns a generator; it can only be iterated once.
    """
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        prompts = [_extract_prompt(record) for record in batch]
        task_ids = [
            str(record["task_id"]) if "task_id" in record else None for record in batch
        ]
        prompt_ids = [
            (
                str(record["prompt_id"])
                if "prompt_id" in record
                else (str(record["id"]) if "id" in record else None)
            )
            for record in batch
        ]
        yield {
            "prompts": prompts,
            "task_ids": task_ids,
            "prompt_ids": prompt_ids,
        }


def _load_reward_mixer(path: Path) -> Dict[str, float]:
    try:
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Failed to parse reward mixer at {path}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"Reward mixer config at {path} must be a mapping")
    normalized: Dict[str, float] = {}
    for key, value in data.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cannot convert reward mixer key '{key}' to float: {value}"
            ) from exc
    return normalized


def _load_mapping_config(path: Path) -> FeedbackMappingConfig:
    try:
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Failed to parse mapping config at {path}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"Mapping config at {path} must be a mapping")
    reward_keys = data.get("reward_keys", {})
    tag_keys = data.get("tag_keys", {})
    if not isinstance(reward_keys, dict):
        raise TypeError(f"mapping_config.reward_keys at {path} must be a mapping")
    if not isinstance(tag_keys, dict):
        raise TypeError(f"mapping_config.tag_keys at {path} must be a mapping")
    task_id_field = data.get("task_id_field", "task_id")
    prompt_id_field = data.get("prompt_id_field", "prompt_id")
    abstain_field = data.get("abstain_field")
    if not isinstance(task_id_field, str):
        raise TypeError(f"mapping_config.task_id_field at {path} must be a string")
    if not isinstance(prompt_id_field, str):
        raise TypeError(f"mapping_config.prompt_id_field at {path} must be a string")
    if abstain_field is not None and not isinstance(abstain_field, str):
        raise TypeError(f"mapping_config.abstain_field at {path} must be a string")
    return FeedbackMappingConfig(
        reward_keys={str(k): str(v) for k, v in reward_keys.items()},
        tag_keys={str(k): str(v) for k, v in tag_keys.items()},
        task_id_field=str(task_id_field),
        prompt_id_field=str(prompt_id_field),
        abstain_field=abstain_field,
    )


def _resolve_dtype(name: str | None) -> Any:
    if not name:
        return None
    if not hasattr(torch, name):
        raise TypeError(f"{_UNSUPPORTED_DTYPE_ERROR}: {name}")
    return getattr(torch, name)


def _build_policy(
    model_name: str,
    device: str,
    dtype: Any,
    device_map: str | None,
    trust_remote_code: bool,
) -> HFPolicyAdapter:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    if device_map is None:
        model.to(device)
    adapter_device = device if device_map is None else _resolve_adapter_device(model)
    return HFPolicyAdapter(model=model, tokenizer=tokenizer, device=adapter_device)


def _resolve_adapter_device(model: Any) -> str | None:
    device_map = getattr(model, "hf_device_map", None) or getattr(
        model, "device_map", None
    )
    if isinstance(device_map, dict):
        for value in device_map.values():
            if isinstance(value, str) and value != "cpu":
                return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a DAPO hybrid policy on RG tasks."
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--dataset", required=True, help="Path to a JSONL dataset")
    parser.add_argument("--output-dir", required=True, help="Output directory for logs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--kl-target", type=float, default=0.1)
    parser.add_argument("--kl-coef", type=float, default=0.1)
    parser.add_argument(
        "--reward-mixer", required=True, help="YAML/JSON reward mixer file"
    )
    parser.add_argument(
        "--mapping-config",
        help="YAML/JSON feedback mapping config",
    )
    parser.add_argument("--enable-grn-policy", action="store_true")
    parser.add_argument("--enable-grn-value", action="store_true")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Maximum training steps"
    )
    parser.add_argument("--torch-dtype", help="Torch dtype (e.g., float16, bfloat16)")
    parser.add_argument("--device-map", help="Transformers device map (e.g., auto)")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code when loading from Hugging Face.",
    )
    args = parser.parse_args()

    device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
    dtype = _resolve_dtype(args.torch_dtype)
    policy = _build_policy(
        args.model,
        device,
        dtype,
        args.device_map,
        args.trust_remote_code,
    )

    reward_weights = _load_reward_mixer(Path(args.reward_mixer))
    reward_mixer = RewardMixerConfig(weights=reward_weights)

    dapo_cfg = DAPOConfig(
        learning_rate=args.learning_rate,
        clip_ratio=args.clip_ratio,
        kl_target=args.kl_target,
        kl_coef=args.kl_coef,
    )
    grn_cfg = GRNConfig(
        enable_policy=args.enable_grn_policy,
        enable_value=args.enable_grn_value,
        enable_probes=False,
    )

    if args.mapping_config:
        feedback_cfg = _load_mapping_config(Path(args.mapping_config))
    else:
        feedback_cfg = FeedbackMappingConfig(reward_keys={}, tag_keys={})

    dataset_rows = _load_jsonl(Path(args.dataset))
    dataloader = _batch_records(dataset_rows, args.batch_size)

    logger = JSONLLogger(Path(args.output_dir) / "train.jsonl")
    trainer = DAPOHybridTrainer(
        policy=policy,
        scorer=NullScorer(),
        dataloader=dataloader,
        feedback_cfg=feedback_cfg,
        cfg=HybridTrainingConfig(
            dapo=dapo_cfg,
            grn=grn_cfg,
            reward_mixer=reward_mixer,
            group_size=args.group_size,
            eval_every=args.eval_every,
            seed=args.seed,
            max_steps=args.max_steps,
        ),
        logger=logger,
        curriculum=CurriculumTracker(),
        safety=SafetyController(),
    )
    trainer.run()


if __name__ == "__main__":
    main()
