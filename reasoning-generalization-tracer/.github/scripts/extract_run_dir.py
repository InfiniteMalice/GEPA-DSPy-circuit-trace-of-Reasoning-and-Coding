#!/usr/bin/env python3
"""Extract run_dir from rg-tracer JSON output."""
from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as exc:  # pragma: no cover - CI safety
        msg = f"Failed to parse rg-tracer output: {exc}"
        print(msg, file=sys.stderr)
        return 1
    run_dir = data.get("run_dir")
    if not isinstance(run_dir, str) or not run_dir:
        print(f"Missing run_dir in output: {data}", file=sys.stderr)
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
