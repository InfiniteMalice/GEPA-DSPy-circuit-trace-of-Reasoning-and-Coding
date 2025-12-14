"""Epistemic-grounded thought alignment scoring."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ..scoring import aggregator


_REASONING_CUES = ("therefore", "because", "hence", "so", "thus", "as a result")
_UNCERTAINTY_STABILISERS = ("probably", "likely", "seems", "suggests")
_RANDOMNESS_FLAGS = ("random", "guess", "no idea", "unsure", "confused")
_CONTRADICTION_FLAGS = ("contradiction", "inconsistent", "but then", "however")
_VACILLATION_PATTERNS = ("or maybe", "or perhaps", "alternatively")


def _collect_text(blob: Any) -> str:
    parts: list[str] = []
    seen: set[int] = set()

    def _walk(value: Any) -> None:
        if value is None:
            return
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        if isinstance(value, str):
            if value.strip():
                parts.append(value)
            return
        if isinstance(value, Mapping):
            for _, entry in sorted(value.items(), key=lambda item: str(item[0])):
                _walk(entry)
            return
        if isinstance(value, (list, tuple)):
            for entry in value:
                _walk(entry)
            return
        if isinstance(value, (set, frozenset)):
            for entry in sorted(value, key=lambda item: str(item)):
                _walk(entry)
            return
        parts.append(str(value))

    _walk(blob)
    return " ".join(parts)


def _normalise_text(trace: Any, context: Any = None) -> str:
    trace_text = _collect_text(trace)
    context_text = _collect_text(context) if context is not None else ""
    combined = " ".join([trace_text, context_text]).strip().casefold()
    return re.sub(r"\s+", " ", combined)


def compute_match_score(trace: Any, answer: Any, context: Any | None = None) -> float:
    """Score whether the trace converges on ``answer`` and prunes alternatives.

    High scores require a derivation plus explicit endorsement of the proposed answer. Penalties
    capture dangling candidates and unresolved contradictions.
    """

    text = _normalise_text(trace, context)
    if not text:
        return 0.0

    score = 0.25  # base credit for providing any trace
    answer_token = str(answer).strip().casefold()
    derivation_found = False
    endorsement_found = False

    if answer_token:
        derivation_pattern = rf"(=|->|=>|yields|gives)\s*{re.escape(answer_token)}"
        if re.search(derivation_pattern, text):
            score += 0.35
            derivation_found = True
        if re.search(rf"(therefore|thus|so).*{re.escape(answer_token)}", text):
            score += 0.25
            endorsement_found = True
        if answer_token in text and not (derivation_found or endorsement_found):
            score += 0.1
    else:
        score -= 0.1

    unresolved_candidates = text.count(" or ") + text.count(" maybe ")
    score -= 0.1 * unresolved_candidates

    if any(flag in text for flag in _CONTRADICTION_FLAGS):
        score -= 0.2

    score = max(0.0, min(1.0, score))
    return score


def compute_epistemic_score(trace: Any) -> float:
    """Measure epistemic grounding of a trace independent of correctness."""

    text = _normalise_text(trace)
    if not text:
        return 0.0

    score = 0.3
    for cue in _REASONING_CUES:
        if cue in text:
            score += 0.12
    for stabiliser in _UNCERTAINTY_STABILISERS:
        if stabiliser in text:
            score += 0.05

    if any(flag in text for flag in _RANDOMNESS_FLAGS):
        score -= 0.35
    if any(flag in text for flag in _CONTRADICTION_FLAGS):
        score -= 0.2
    if any(pattern in text for pattern in _VACILLATION_PATTERNS):
        score -= 0.15

    score = max(0.0, min(1.0, score))
    return score


def _get_thresholds() -> tuple[float, float]:
    cfg = aggregator.get_last_config() or {}
    raw_ta_cfg = cfg.get("thought_alignment", {}) if isinstance(cfg, Mapping) else {}
    ta_cfg = raw_ta_cfg if isinstance(raw_ta_cfg, Mapping) else {}
    try:
        theta_match = float(ta_cfg.get("theta_match", 0.8))
    except (TypeError, ValueError):
        theta_match = 0.8
    try:
        theta_epistemic = float(ta_cfg.get("theta_epistemic", 0.5))
    except (TypeError, ValueError):
        theta_epistemic = 0.5
    theta_match = max(0.0, min(1.0, theta_match))
    theta_epistemic = max(0.0, min(1.0, theta_epistemic))
    return theta_match, theta_epistemic


def classify_thought_alignment(
    trace: Any,
    answer: Any,
    context: Any | None = None,
    *,
    thresholds: tuple[float, float] | None = None,
) -> tuple[bool, float, float]:
    """Classify thought alignment using match and epistemic thresholds."""

    s_match = compute_match_score(trace, answer, context)
    s_epistemic = compute_epistemic_score(trace)
    resolved = thresholds if thresholds is not None else _get_thresholds()
    if not (isinstance(resolved, tuple) and len(resolved) == 2):
        raise ValueError("thresholds must be a (theta_match, theta_epistemic) tuple")
    theta_match, theta_epistemic = (float(resolved[0]), float(resolved[1]))
    theta_match = max(0.0, min(1.0, theta_match))
    theta_epistemic = max(0.0, min(1.0, theta_epistemic))
    thought_align = s_match >= theta_match and s_epistemic >= theta_epistemic
    return thought_align, s_match, s_epistemic


__all__ = [
    "classify_thought_alignment",
    "compute_epistemic_score",
    "compute_match_score",
]
