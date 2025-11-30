#!/usr/bin/env python3
"""Validate attribution artifacts emitted by CI smoke tests."""
from __future__ import annotations

import json
import os
from pathlib import Path


REQUIRED_KEYS: tuple[str, ...] = (
    "delta_alignment",
    "delta_repeatability",
    "delta_sparsity",
)


def _load_lines(metrics_path: Path) -> list[dict[str, object]]:
    text = metrics_path.read_text(encoding="utf8")
    records: list[dict[str, object]] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:  # pragma: no cover - CI safety
            raise SystemExit(f"Failed to parse metrics line {raw!r}: {exc}") from exc
        if isinstance(data, dict):
            records.append(data)
    return records


def main() -> int:
    run_dir_raw = os.environ.get("RUN_DIR")
    if not run_dir_raw:
        raise SystemExit("RUN_DIR environment variable is required")
    run_dir = Path(run_dir_raw)
    metrics_path = run_dir / "attr_metrics.jsonl"
    attr_dir = run_dir / "attr"
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics file: {metrics_path}")
    lines = _load_lines(metrics_path)
    if not lines:
        raise SystemExit("attr_metrics.jsonl is empty")
    first = lines[0]
    missing = [key for key in REQUIRED_KEYS if key not in first]
    if missing:
        raise SystemExit(f"Missing keys {missing} in metrics: {first}")
    if not attr_dir.exists():
        raise SystemExit(f"Missing attr directory: {attr_dir}")
    entries = list(attr_dir.iterdir())
    if not entries:
        raise SystemExit("attr directory empty")
    if not any(path.suffix == ".json" for path in entries):
        raise SystemExit("attr directory lacks JSON files")
    summary_path = run_dir / "summary.md"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary: {summary_path}")
    if "Attribution Metrics" not in summary_path.read_text(encoding="utf8"):
        raise SystemExit("summary.md missing attribution section")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
