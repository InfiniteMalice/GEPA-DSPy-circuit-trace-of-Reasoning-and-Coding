"""Tests for the additive schema V3 trajectory overlay."""

from __future__ import annotations

import json

from rg_tracer.schema_v3 import TrajectoryOverlay, classify_case_v3


def test_trajectory_overlay_serializes_without_changing_case_id():
    result = classify_case_v3(
        output_text="5",
        expected_answer="5",
        is_idk=False,
        confidence=0.9,
        thought_aligned=True,
        trajectory_overlay=TrajectoryOverlay(
            sampler_name="gram_mdt",
            total_updates=4,
            max_depth=2,
            max_width=3,
            branch_count=3,
            active_views=["arithmetic"],
            view_route=["arithmetic", "verification"],
            process_score_components={"confidence": 0.9},
        ),
    )
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["case_id"] == 1
    assert payload["trajectory_overlay"]["sampler_name"] == "gram_mdt"
