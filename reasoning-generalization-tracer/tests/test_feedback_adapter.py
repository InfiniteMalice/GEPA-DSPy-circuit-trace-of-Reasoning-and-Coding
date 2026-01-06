"""Tests for GEPA feedback mapping adapter."""

from __future__ import annotations

from rg_tracer.dapo.feedback_adapter import FeedbackMappingConfig, make_gepa_feedback


def test_make_gepa_feedback_mapping() -> None:
    cfg = FeedbackMappingConfig(
        reward_keys={"correct": "correctness"},
        tag_keys={"fmt": "format_penalty"},
        abstain_field="abstain",
    )
    feedback = make_gepa_feedback(
        prompt="p",
        completion="c",
        local_metrics={"correct": 1.0, "fmt": 0.2, "abstain": 1.0},
        meta={"task_id": "t1", "prompt_id": "p1"},
        cfg=cfg,
    )

    assert feedback.rewards == {"correctness": 1.0}
    assert feedback.tags == {"format_penalty": 0.2}
    assert feedback.meta["task_id"] == "t1"
    assert feedback.meta["prompt_id"] == "p1"
    assert feedback.abstained is True
