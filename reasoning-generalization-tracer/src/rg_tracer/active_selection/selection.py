"""Select diagnostic tasks that distinguish candidate hypotheses."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from .disagreement import disagreement_score


def select_diagnostic_tasks(
    task_predictions: Iterable[Mapping[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    scored: List[Dict[str, Any]] = []
    for item in task_predictions:
        predictions = [str(value) for value in item.get("predictions", [])]
        score = disagreement_score(predictions)
        reason = "candidate predictions differ" if score > 0 else "candidate predictions agree"
        scored.append(
            {
                "task_id": item.get("task_id"),
                "disagreement_score": score,
                "metadata": {
                    "reason": reason,
                    "candidate_count": len(predictions),
                },
            }
        )
    scored.sort(key=lambda row: (-float(row["disagreement_score"]), str(row["task_id"])))
    return scored[:top_k]


__all__ = ["select_diagnostic_tasks"]
