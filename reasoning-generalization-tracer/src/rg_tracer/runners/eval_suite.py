"""Batch evaluation utilities for datasets."""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping

from ..scoring import axes, aggregator

AXIS_FUNCTIONS = {name: getattr(axes, name) for name in axes.__all__}


def _load_records(pattern: str) -> List[Mapping[str, object]]:
    paths = sorted(glob.glob(pattern))
    records: List[Mapping[str, object]] = []
    for path in paths:
        with open(path, "r", encoding="utf8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def evaluate_dataset(
    dataset_pattern: str,
    profile: str,
    *,
    output_csv: str | Path | None = None,
) -> Dict[str, object]:
    records = _load_records(dataset_pattern)
    profiles = aggregator.load_profiles()
    if profile not in profiles:
        raise KeyError(f"Profile {profile} not found")
    profile_obj = profiles[profile]

    per_axis_scores: Dict[str, List[int]] = {axis: [] for axis in AXIS_FUNCTIONS}
    record_axis_scores: List[Dict[str, int]] = []
    composites: List[float] = []
    gate_passes = 0

    for record in records:
        metrics = record.get("metrics", {})
        scores = {axis: func(metrics.get(axis, {})) for axis, func in AXIS_FUNCTIONS.items()}
        for axis, score in scores.items():
            per_axis_scores[axis].append(score)
        eval_result = aggregator.evaluate_profile(scores, profile_obj)
        composites.append(float(eval_result["composite"]))
        gate_passes += int(eval_result["passes_gates"])
        record_axis_scores.append(scores)

    summary = {
        "axis_means": {
            axis: mean(values) if values else 0.0 for axis, values in per_axis_scores.items()
        },
        "composite_mean": mean(composites) if composites else 0.0,
        "gate_pass_rate": gate_passes / len(records) if records else 0.0,
        "count": len(records),
    }

    if output_csv is not None:
        fieldnames = ["record_id", "composite"] + [f"axis_{axis}" for axis in AXIS_FUNCTIONS]
        with open(output_csv, "w", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record, scores, composite in zip(records, record_axis_scores, composites):
                row = {"record_id": record.get("id"), "composite": composite}
                for axis in AXIS_FUNCTIONS:
                    row[f"axis_{axis}"] = scores.get(axis, 0)
                writer.writerow(row)

    return summary


__all__ = ["evaluate_dataset"]
