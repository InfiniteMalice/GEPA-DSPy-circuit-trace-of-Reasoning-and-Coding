from __future__ import annotations


def build_taxonomy(supported: int, contradicted: int) -> dict:
    primary = "factual_error" if contradicted else "unsupported_bridge_inference"
    axis_ie = "extrinsic" if supported == 0 else "mixed"
    axis_ff = "factuality" if contradicted or supported == 0 else "neither"
    return {
        "hallucination_axis_intrinsic_extrinsic": axis_ie,
        "hallucination_axis_factuality_faithfulness": axis_ff,
        "hallucination_primary_type": primary,
        "hallucination_secondary_types": [],
        "task_specific_hallucination_type": "qa",
    }
