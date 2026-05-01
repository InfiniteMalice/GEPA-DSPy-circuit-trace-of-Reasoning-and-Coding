from __future__ import annotations


def infer_observability_tier(
    has_telemetry: bool, has_external_verification: bool, has_trace: bool
) -> str:
    if has_trace and has_external_verification:
        return "O5"
    if has_external_verification:
        return "O3"
    if has_telemetry:
        return "O2"
    return "O0"
