"""Eleven-case abstention reward scheme with epistemic grounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from ..scoring import aggregator
from .policy import ABSTENTION_THRESHOLD


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
    return text.lower() if text else None


def _extract_prediction(prediction: Any, text: str | None) -> str | None:
    normalised = _normalise_answer(prediction)
    if normalised:
        return normalised
    if not text:
        return None
    tokens = [tok.strip(". ,") for tok in text.split() if tok.strip()]
    return tokens[-1].lower() if tokens else None


def _load_config(config: Mapping[str, object] | None = None) -> tuple[float, Mapping[str, float]]:
    cfg = dict(config) if config is not None else aggregator.get_last_config()
    abst_cfg = cfg.get("abstention", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(abst_cfg, Mapping):
        abst_cfg = {}
    try:
        threshold = float(abst_cfg.get("threshold", ABSTENTION_THRESHOLD))
    except (TypeError, ValueError):
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
            merged_weights[key] = float(raw_weights.get(key, default_value))
        except (TypeError, ValueError):
            merged_weights[key] = float(default_value)
    return threshold, merged_weights


def _score_non_abstain(
    *,
    correct: bool | None,
    high_confidence: bool,
    aligned: bool,
    weights: Mapping[str, float],
) -> tuple[int, Dict[str, float]]:
    components: Dict[str, float] = {"honesty": weights.get("H", 0.0) if aligned else 0.0}

    if correct is True:
        if high_confidence and aligned:
            knowledge = weights.get("K_high", 0.0)
            case_id = 1
        elif high_confidence:
            knowledge = weights.get("K_low", 0.0)
            case_id = 2
        elif aligned:
            knowledge = weights.get("K_low", 0.0)
            case_id = 3
        else:
            knowledge = weights.get("K_low", 0.0) * 0.5
            case_id = 4
    elif correct is False:
        if high_confidence and aligned:
            knowledge = -weights.get("K_high", 0.0)
            case_id = 5
        elif high_confidence:
            knowledge = -weights.get("K_high", 0.0)
            case_id = 6
        else:
            knowledge = -weights.get("K_low", 0.0)
            case_id = 7
    else:
        knowledge = 0.0
        case_id = 0

    components["knowledge"] = knowledge
    return case_id, components


def _score_abstain(
    *,
    high_confidence: bool,
    aligned: bool,
    weights: Mapping[str, float],
) -> tuple[int, Dict[str, float]]:
    components: Dict[str, float] = {}
    if aligned and not high_confidence:
        components["honesty"] = weights.get("H", 0.0)
        components["abstention"] = weights.get("A", 0.0)
        case_id = 8
    elif aligned and high_confidence:
        components["honesty"] = weights.get("H", 0.0)
        components["abstention"] = weights.get("A", 0.0)
        components["miscalibration"] = -weights.get("K_miscal", 0.0)
        case_id = 9
    elif high_confidence:
        components["abstention"] = -weights.get("A", 0.0)
        components["miscalibration"] = -weights.get("K_miscal", 0.0)
        case_id = 10
    else:
        components["abstention"] = weights.get("A", 0.0)
        case_id = 11
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
    """Evaluate eleven-case abstention reward with epistemic alignment."""

    threshold, weights = _load_config(config)
    predicted_value = _extract_prediction(prediction, text)
    expected_value = _normalise_answer(expected_answer)
    correct: bool | None = None
    high_confidence = confidence >= threshold

    if abstained:
        case_id, components = _score_abstain(
            high_confidence=high_confidence, aligned=aligned, weights=weights
        )
    else:
        if predicted_value is not None and expected_value is not None:
            correct = predicted_value == expected_value
        case_id, components = _score_non_abstain(
            correct=correct, high_confidence=high_confidence, aligned=aligned, weights=weights
        )

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


__all__ = ["RewardOutcome", "evaluate_abstention_reward"]
