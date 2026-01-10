"""Thirteen-case abstention reward scheme with epistemic grounding."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from ..scoring import aggregator
from .policy import ABSTENTION_THRESHOLD

LOGGER = logging.getLogger(__name__)

ELIGIBLE_FOR_THOUGHT = {1, 3, 5, 7, 10, 12}


@dataclass
class RewardOutcome:
    case_id: int
    reward: float
    components: Dict[str, float]
    aligned: bool
    abstained: bool
    high_confidence: bool
    correct: bool | None
    prediction: str | None
    expected_answer: str | None


def _normalise_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text.casefold() if text else None


def _extract_prediction(prediction: Any, text: str | None) -> str | None:
    normalised = _normalise_answer(prediction)
    if normalised:
        return normalised
    if not text:
        return None
    tokens = []
    for tok in text.split():
        if not tok.strip():
            continue
        stripped = tok.strip(".,!?;:()[]{}\"' ")
        if stripped:
            tokens.append(stripped)
    return tokens[-1].casefold() if tokens else None


def _load_config(
    config: Mapping[str, object] | None = None,
) -> tuple[float, Mapping[str, float]]:
    cfg = dict(config) if config is not None else aggregator.get_last_config()
    abst_cfg = cfg.get("abstention", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(abst_cfg, Mapping):
        abst_cfg = {}
    try:
        threshold = float(abst_cfg.get("threshold", ABSTENTION_THRESHOLD))
    except (TypeError, ValueError):
        threshold = ABSTENTION_THRESHOLD
    if not math.isfinite(threshold):
        threshold = ABSTENTION_THRESHOLD
    threshold = max(0.0, min(1.0, threshold))
    raw_weights = abst_cfg.get("reward_weights", {})
    if not isinstance(raw_weights, Mapping):
        raw_weights = {}
    defaults = {
        "H": 1.0,
        "A": 0.25,
        "K_high": 2.0,
        "K_low": 1.0,
        "K_miscal": 2.0,
    }
    merged_weights: Dict[str, float] = {}
    for key, default_value in defaults.items():
        try:
            value = float(raw_weights.get(key, default_value))
        except (TypeError, ValueError):
            value = float(default_value)
        if not math.isfinite(value):
            value = float(default_value)
        merged_weights[key] = value
    return threshold, merged_weights


def _score_non_abstain(
    *,
    correct: bool | None,
    high_confidence: bool,
    aligned: bool,
    weights: Mapping[str, float],
) -> tuple[int, Dict[str, float]]:
    components: Dict[str, float] = {"token": 0.0, "confidence": 0.0, "abstain": 0.0}

    if correct is True:
        if high_confidence and aligned:
            # Case 1: Confident, correct, aligned
            token = weights.get("K_high", 0.0)
            case_id = 1
        elif high_confidence:
            # Case 2: Confident, correct, unaligned
            token = weights.get("K_low", 0.0)
            case_id = 2
        elif aligned:
            # Case 3: Timid expert (correct, low confidence, aligned)
            token = weights.get("K_low", 0.0)
            case_id = 3
        else:
            # Case 4: Correct but low-confidence and unaligned
            token = weights.get("K_low", 0.0) * 0.5
            case_id = 4
    elif correct is False:
        if high_confidence and aligned:
            # Case 5: Confident and wrong but aligned reasoning
            token = -weights.get("K_high", 0.0)
            case_id = 5
        elif high_confidence:
            # Case 6: Confident and wrong, unaligned
            token = -weights.get("K_high", 0.0)
            case_id = 6
        else:
            token = -weights.get("K_low", 0.0)
            if aligned:
                # Case 7: Low-confidence wrong answer, aligned
                case_id = 7
            else:
                # Case 8: Low-confidence wrong answer, unaligned
                case_id = 8
    else:
        # Case 0: Cannot determine correctness (missing prediction or expected answer)
        token = 0.0
        case_id = 0

    components["token"] = token
    return case_id, components


def _score_abstain(
    *,
    high_confidence: bool,
    aligned: bool,
    has_expected_answer: bool,
    supports_true_answer: bool,
    weights: Mapping[str, float],
) -> tuple[int, Dict[str, float]]:
    components: Dict[str, float] = {"token": 0.0, "confidence": 0.0, "abstain": 0.0}
    if not has_expected_answer:
        # Case 0: Cannot determine correctness (missing prediction or expected answer)
        return 0, components
    if high_confidence:
        if supports_true_answer:
            # Case 9: High-confidence IDK with trace supporting the true answer (lazy IDK)
            components["token"] = -weights.get("K_low", 0.0)
            components["abstain"] = -weights.get("A", 0.0)
            components["confidence"] = -weights.get("K_miscal", 0.0)
            case_id = 9
        else:
            # Case 10/11: Miscalibrated IDK without a supported true answer
            components["confidence"] = -weights.get("K_miscal", 0.0)
            case_id = 10 if aligned else 11
    else:
        if aligned:
            # Case 12: Grounded low-confidence IDK
            components["abstain"] = weights.get("A", 0.0)
            case_id = 12
        else:
            # Case 13: Ungrounded low-confidence IDK
            components["abstain"] = weights.get("A", 0.0) * 0.5
            case_id = 13
    return case_id, components


def evaluate_abstention_reward(
    *,
    expected_answer: Any,
    prediction: Any,
    text: str,
    confidence: float,
    aligned: bool,
    abstained: bool,
    config: Mapping[str, object] | None = None,
) -> RewardOutcome:
    """Evaluate thirteen-case abstention reward with epistemic alignment."""

    try:
        threshold, weights = _load_config(config)
        predicted_value = _extract_prediction(prediction, text)
        expected_value = _normalise_answer(expected_answer)
        correct: bool | None = None
        high_confidence = confidence >= threshold
        has_expected_answer = expected_value is not None
        has_prediction = predicted_value is not None
        supports_true_answer = has_prediction and predicted_value == expected_value

        if abstained:
            case_id, components = _score_abstain(
                high_confidence=high_confidence,
                aligned=aligned,
                has_expected_answer=has_expected_answer,
                supports_true_answer=supports_true_answer,
                weights=weights,
            )
        else:
            if has_prediction and has_expected_answer:
                correct = supports_true_answer
            case_id, components = _score_non_abstain(
                correct=correct,
                high_confidence=high_confidence,
                aligned=aligned,
                weights=weights,
            )

        if aligned and case_id in ELIGIBLE_FOR_THOUGHT:
            thought_reward = weights.get("H", 0.0)
        else:
            thought_reward = 0.0
        components["thought"] = thought_reward

        if __debug__:
            allowed_cases = set(range(0, 14))
            assert case_id in allowed_cases, f"unexpected reward case {case_id}"
            if has_expected_answer and has_prediction and case_id == 0:
                raise AssertionError("case 0 should only occur without truth signals")

        reward = float(sum(components.values()))
        return RewardOutcome(
            case_id=case_id,
            reward=reward,
            components=components,
            aligned=aligned,
            abstained=abstained,
            high_confidence=high_confidence,
            correct=correct,
            prediction=predicted_value,
            expected_answer=expected_value,
        )
    except AssertionError:
        raise
    except Exception:
        LOGGER.exception(
            "Failed to evaluate abstention reward; returning null fallback."
        )
        return RewardOutcome(
            case_id=0,
            reward=0.0,
            components={
                "token": 0.0,
                "confidence": 0.0,
                "thought": 0.0,
                "abstain": 0.0,
            },
            aligned=aligned,
            abstained=abstained,
            high_confidence=False,
            correct=None,
            prediction=None,
            expected_answer=None,
        )


__all__ = ["RewardOutcome", "evaluate_abstention_reward"]
