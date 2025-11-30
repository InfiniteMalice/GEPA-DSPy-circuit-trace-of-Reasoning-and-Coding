"""Confidence calibration helpers using only the standard library."""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Tuple


def _clip(prob: float) -> float:
    return min(max(prob, 1e-6), 1 - 1e-6)


def temperature_scale(
    confidences: Iterable[float], labels: Iterable[int]
) -> Callable[[float], float]:
    """Fit a temperature scaling function on binary confidences."""
    confidences = [_clip(float(c)) for c in confidences]
    labels = [int(label_value) for label_value in labels]
    if len(confidences) != len(labels):
        raise ValueError("confidences and labels length mismatch")
    logits = [math.log(c / (1 - c)) for c in confidences]
    temps = [0.5 + i * 0.05 for i in range(1, 101)]
    best_temp = 1.0
    best_loss = float("inf")

    for temp in temps:
        loss = 0.0
        for logit, label in zip(logits, labels):
            scaled = 1 / (1 + math.exp(-logit / temp))
            loss -= label * math.log(_clip(scaled)) + (1 - label) * math.log(_clip(1 - scaled))
        loss /= max(1, len(labels))
        if loss < best_loss:
            best_loss = loss
            best_temp = temp

    def calibrate(conf: float) -> float:
        conf = _clip(conf)
        logit = math.log(conf / (1 - conf))
        return 1 / (1 + math.exp(-logit / best_temp))

    return calibrate


def isotonic_calibration(
    confidences: Iterable[float], labels: Iterable[int]
) -> Callable[[float], float]:
    """Fit an isotonic regression calibrator using the PAV algorithm."""
    confidences_list = [float(conf) for conf in confidences]
    labels_list = [int(label_value) for label_value in labels]
    if len(confidences_list) != len(labels_list):
        raise ValueError("confidences and labels length mismatch")
    pairs = sorted(zip(confidences_list, labels_list))
    if not pairs:
        return lambda conf: conf

    # Pool-adjacent-violators: maintain blocks with monotonic means
    blocks: List[Tuple[float, float, float]] = []  # (weight, sum, mean)
    for conf, label in pairs:
        weight = 1.0
        total = label
        mean = total / weight
        blocks.append((weight, total, mean))
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            w1, s1, _ = blocks[-2]
            w2, s2, _ = blocks[-1]
            new_weight = w1 + w2
            new_sum = s1 + s2
            new_mean = new_sum / new_weight
            blocks = blocks[:-2] + [(new_weight, new_sum, new_mean)]

    thresholds: List[float] = []
    values: List[float] = []
    start = 0
    for weight, total, mean in blocks:
        end = start + int(round(weight))
        thresholds.extend([pairs[start][0], pairs[end - 1][0]])
        values.append(mean)
        start = end

    def calibrate(conf: float) -> float:
        conf = float(conf)
        if conf <= thresholds[0]:
            return values[0]
        if conf >= thresholds[-1]:
            return values[-1]
        for idx in range(len(values)):
            lower = thresholds[2 * idx]
            upper = thresholds[2 * idx + 1]
            if lower <= conf <= upper:
                return values[idx]
        return values[-1]

    return calibrate


__all__ = ["temperature_scale", "isotonic_calibration"]
