from __future__ import annotations

import hashlib
import json


_RESERVED = {
    "trace_package_id",
    "graph_candidate_priority",
    "suspected_failure_mode",
    "supported_trace_artifacts",
    "downstream_graph_status",
}


def build_trace_package(payload: dict) -> dict:
    clean_payload = {k: v for k, v in payload.items() if k not in _RESERVED}
    blob = json.dumps(clean_payload, sort_keys=True)
    trace_id = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    result = dict(clean_payload)
    result["trace_package_id"] = trace_id
    result["graph_candidate_priority"] = clean_payload.get("hallucination_risk", 0.0)
    result["suspected_failure_mode"] = clean_payload.get("suspected_failure_mode", "unknown")
    result["supported_trace_artifacts"] = ["atomic-facts", "evidence-map"]
    result["downstream_graph_status"] = "queued"
    return result
