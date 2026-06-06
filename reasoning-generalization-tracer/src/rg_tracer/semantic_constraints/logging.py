"""Structured logging for semantic constraint compilation."""

from __future__ import annotations

import json
from pathlib import Path

from .types import SemanticProjectionResult


def write_semantic_constraint_logs(
    output_dir: Path,
    projection: SemanticProjectionResult,
) -> None:
    """Write JSONL and summary logs without private chain-of-thought."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = projection.as_dict()
    with (output_dir / "semantic_constraints.jsonl").open("a", encoding="utf8") as handle:
        handle.write(json.dumps(payload) + "\n")
    summary = {
        "input_text": projection.compilation.input_text,
        "projection_mode": projection.mode,
        "compiled_constraint_count": len(projection.compilation.constraints),
        "verified_constraint_count": projection.compilation.diagnostics.get(
            "verified_constraint_count",
            0,
        ),
        "unsupported_fragments": projection.compilation.unsupported_fragments,
        "ambiguity_detected": projection.compilation.ambiguity_detected,
        "contradiction_detected": projection.compilation.contradiction_detected,
        "verification_status": projection.compilation.verification_status,
        "remaining_candidates": projection.remaining_candidates,
        "recommended_action": projection.recommended_action,
    }
    with (output_dir / "semantic_constraint_summary.json").open("w", encoding="utf8") as handle:
        json.dump(summary, handle, indent=2)


__all__ = ["write_semantic_constraint_logs"]
