from __future__ import annotations

import hashlib
import json


def build_trace_package(payload: dict) -> dict:
    blob = json.dumps(payload, sort_keys=True)
    trace_id = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return {
        "trace_package_id": trace_id,
        "graph_candidate_priority": payload.get("hallucination_risk", 0.0),
        "suspected_failure_mode": payload.get("suspected_failure_mode", "unknown"),
        "supported_trace_artifacts": ["atomic-facts", "evidence-map"],
        "downstream_graph_status": "queued",
        **payload,
    }
