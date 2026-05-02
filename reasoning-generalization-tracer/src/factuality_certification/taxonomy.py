from __future__ import annotations


def build_taxonomy(supported: int, contradicted: int, unsupported: int) -> dict:
    if contradicted > 0:
        primary = "factual_error"
        axis_ie = "mixed"
        axis_ff = "factuality"
    elif supported > 0 and unsupported == 0:
        primary = "clean_support"
        axis_ie = "intrinsic"
        axis_ff = "neither"
    elif supported > 0:
        primary = "partially_supported"
        axis_ie = "mixed"
        axis_ff = "unknown"
    else:
        primary = "unsupported_bridge_inference"
        axis_ie = "extrinsic"
        axis_ff = "factuality"
    return {
        "hallucination_axis_intrinsic_extrinsic": axis_ie,
        "hallucination_axis_factuality_faithfulness": axis_ff,
        "hallucination_primary_type": primary,
        "hallucination_secondary_types": [],
        "task_specific_hallucination_type": "qa",
    }
