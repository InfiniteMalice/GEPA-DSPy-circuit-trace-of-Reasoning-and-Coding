"""Artifact helpers for optional LRD grokking experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def json_safe(value: Any) -> Any:
    """Convert dataclasses and tensors into JSON-safe objects."""

    if is_dataclass(value):
        return json_safe(asdict(value))
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        json.dump(json_safe(payload), handle, indent=2)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(json_safe(payload)) + "\n")


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row)) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))


def write_summary_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "mode",
        "seed",
        "train_fraction",
        "memorization_epoch",
        "generalization_epoch",
        "grokking_delay",
        "final_train_accuracy",
        "final_test_accuracy",
        "q_effective_rank",
        "k_effective_rank",
        "wall_clock_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            values = [str(row.get(column, "")) for column in columns]
            handle.write("| " + " | ".join(values) + " |\n")


__all__ = [
    "append_jsonl",
    "json_safe",
    "write_json",
    "write_jsonl",
    "write_summary_csv",
    "write_summary_markdown",
]
