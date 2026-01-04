"""Policy adapter for HuggingFace causal LMs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from gepa_dapo_grn.policy_interfaces import Policy

from ..modules.torch_stub import SimpleTensor, torch


@dataclass
class GenerationOutput:
    completions: List[str]
    actions: List[List[int]]
    logprobs: List[float]
    metadata: List[Dict[str, Any]]


class HFPolicyAdapter(Policy):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: Optional[str] = None,
        frozen: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.frozen = frozen
        if frozen:
            self._freeze_model()

    def _freeze_model(self) -> None:
        if hasattr(self.model, "parameters"):
            for param in self.model.parameters():
                param.requires_grad = False

    def clone(self) -> Policy:
        return HFPolicyAdapter(
            self.model, self.tokenizer, device=self.device, frozen=True
        )

    def forward(self, obs: Any) -> Any:
        inputs = self._prepare_inputs(obs)
        outputs = self.model(**inputs)
        if hasattr(outputs, "logits"):
            return outputs.logits
        return outputs

    def logprobs(
        self, obs: Any, actions: Sequence[Sequence[int]] | Sequence[int]
    ) -> Any:
        """Return log-probabilities for the provided actions.

        The actions must not exceed the logits sequence length when logits are 3D.
        """
        logits = self.forward(obs)
        return _gather_logprobs(logits, actions)

    def generate_with_logprobs(
        self,
        prompts: Sequence[str],
        *,
        group_size: int = 1,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        max_new_tokens: int = 64,
    ) -> GenerationOutput:
        if seed is not None and hasattr(torch, "manual_seed"):
            torch.manual_seed(seed)
        prompt_list = list(prompts)
        inputs = self.tokenizer(prompt_list, return_tensors="pt", padding=True)
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        generation = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=group_size,
            output_scores=True,
            return_dict_in_generate=True,
        )
        sequences = generation.sequences
        completions = []
        actions = []
        logprobs = []
        metadata = []
        prompt_lengths = _prompt_lengths(
            inputs.get("input_ids"),
            prompt_list,
            self.tokenizer.pad_token_id,
        )
        if (
            hasattr(generation, "scores")
            and generation.scores
            and hasattr(torch, "stack")
        ):
            score_tensor = torch.stack(generation.scores, dim=1)
            token_logprobs = _log_softmax(score_tensor)
        else:
            token_logprobs = None
        for index in range(len(sequences)):
            prompt_index = index // group_size
            prompt = prompt_list[prompt_index]
            prompt_len = prompt_lengths[prompt_index]
            token_ids = sequences[index][prompt_len:]
            completion = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            completion_tokens = _to_token_ids(token_ids)
            completions.append(completion)
            actions.append(completion_tokens)
            if token_logprobs is not None and completion_tokens:
                token_scores = token_logprobs[index, -len(completion_tokens) :]
                sequence_logprob = _sum_token_logprobs(token_scores, completion_tokens)
                logprobs.append(sequence_logprob)
            else:
                logprobs.append(0.0)
            metadata.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "temperature": temperature,
                    "seed": seed,
                    "logprobs_available": token_logprobs is not None,
                }
            )
        return GenerationOutput(
            completions=completions,
            actions=actions,
            logprobs=logprobs,
            metadata=metadata,
        )

    def _prepare_inputs(self, obs: Any) -> Dict[str, Any]:
        if isinstance(obs, Mapping):
            return dict(obs)
        if isinstance(obs, str):
            obs = [obs]
        if isinstance(obs, Iterable):
            return self.tokenizer(list(obs), return_tensors="pt", padding=True)
        raise UnsupportedObservationError("Unsupported observation format")


class UnsupportedObservationError(TypeError):
    """Raised when observation format is not supported."""


def _log_softmax(logits: Any) -> Any:
    if hasattr(torch, "log_softmax"):
        return torch.log_softmax(logits, dim=-1)
    data = _to_list(logits)
    if not data:
        return torch.tensor([])
    if (
        isinstance(data[0], Sequence)
        and len(data[0]) > 0
        and isinstance(data[0][0], list)
    ):
        return torch.tensor(
            [[_log_softmax_row(step) for step in sequence] for sequence in data]
        )
    return torch.tensor([_log_softmax_row(row) for row in data])


def _log_softmax_row(row: Sequence[float]) -> List[float]:
    if not row:
        return []
    max_val = max(row)
    exp_vals = [math.exp(value - max_val) for value in row]
    denom = sum(exp_vals)
    return [value - max_val - math.log(denom) for value in row]


def _to_list(tensor: Any) -> List[List[float]]:
    if hasattr(tensor, "tolist"):
        return tensor.tolist()
    if isinstance(tensor, SimpleTensor):
        return tensor.tolist()
    return list(tensor)


def _to_token_ids(token_ids: Any) -> List[int]:
    raw = _to_list(token_ids)
    if not raw:
        return []
    if isinstance(raw[0], list):
        raw = raw[0]
    return [int(token_id) for token_id in raw]


def _sum_token_logprobs(token_scores: Any, token_ids: Sequence[int]) -> float:
    gathered = []
    for idx, token_id in enumerate(token_ids):
        gathered.append(token_scores[idx][token_id])
    summed = sum(gathered)
    if hasattr(summed, "item"):
        return float(summed.item())
    return float(summed)


def _prompt_lengths(
    input_ids: Any,
    prompts: Sequence[str],
    pad_token_id: Optional[int],
) -> List[int]:
    if input_ids is None:
        return [len(prompt) for prompt in prompts]
    rows = _to_list(input_ids)
    lengths = []
    for row in rows:
        if pad_token_id is None:
            lengths.append(len(row))
            continue
        length = len(row)
        for idx, value in enumerate(row):
            if int(value) == int(pad_token_id):
                length = idx
                break
        lengths.append(length)
    return lengths


def _gather_logprobs(
    logits: Any,
    actions: Sequence[Sequence[int]] | Sequence[int],
) -> Any:
    """Gather log probabilities for actions.

    For 3D logits, each action sequence length must be <= logits sequence length.
    """
    log_probs = _log_softmax(logits)
    if isinstance(actions, Sequence) and actions and isinstance(actions[0], int):
        action_ids = list(actions)
        if hasattr(log_probs, "ndim") and log_probs.ndim == 3:
            gathered = [
                log_probs[idx, -1, action] for idx, action in enumerate(action_ids)
            ]
        else:
            gathered = [log_probs[idx][action] for idx, action in enumerate(action_ids)]
        return torch.tensor(gathered)
    gathered_sequences = []
    for idx, token_ids in enumerate(actions):
        seq_len = len(log_probs[idx])
        if len(token_ids) > seq_len:
            raise ValueError(
                "Action sequence length exceeds logits sequence length for batch index "
                f"{idx}: {len(token_ids)} > {seq_len}."
            )
        token_scores = []
        for token_index, token_id in enumerate(token_ids):
            token_scores.append(log_probs[idx][token_index][token_id])
        gathered_sequences.append(sum(token_scores))
    return torch.tensor(gathered_sequences)
