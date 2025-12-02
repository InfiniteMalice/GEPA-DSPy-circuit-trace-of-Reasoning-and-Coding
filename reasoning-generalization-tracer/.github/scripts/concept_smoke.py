#!/usr/bin/env python3
"""Smoke test concept reward alignment path."""
from __future__ import annotations

import sys

from rg_tracer.concepts import ConceptSpec, compute_concept_reward


def main() -> int:
    trace = {
        "features": [{"id": "F1", "layer": 1, "importance": 0.7, "tags": ["parity"]}],
        "edges": [],
        "path_lengths": {"mean": 1.0},
    }
    spec = ConceptSpec(name="parity", definition="", expected_substructures=["parity"])
    reward = compute_concept_reward(
        trace,
        spec,
        task_metrics={
            "concept_reuse": 1.0,
            "supporting_tasks": 1,
            "entailed_feature_ids": ["F1"],
        },
    )
    if reward <= 0.3:
        print(f"Unexpected reward {reward}", file=sys.stderr)
        return 1
    aligned = compute_concept_reward(
        trace,
        spec,
        task_metrics={
            "concept_reuse": 1.0,
            "supporting_tasks": 1,
            "entailed_feature_ids": ["F1"],
        },
        alignment=0.5,
    )
    if aligned <= reward:
        print("Alignment scaling did not increase reward", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
