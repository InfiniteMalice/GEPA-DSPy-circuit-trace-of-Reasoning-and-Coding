"""Open-world perturbation utilities for optional robustness testing."""

from .domain_shift import substitute_role_fillers, swap_surface_domain
from .observation_shift import (
    add_redundant_observation,
    add_stale_observation,
    change_value_format,
    inject_anomaly,
)
from .perturbations import PerturbedTask, make_task
from .query_shift import add_irrelevant_detail, introduce_ambiguity, paraphrase_label
from .tool_shift import (
    add_distractor_tool,
    add_optional_field,
    rename_tool,
    reorder_schema_fields,
)

__all__ = [
    "PerturbedTask",
    "add_distractor_tool",
    "add_irrelevant_detail",
    "add_optional_field",
    "add_redundant_observation",
    "add_stale_observation",
    "change_value_format",
    "inject_anomaly",
    "introduce_ambiguity",
    "make_task",
    "paraphrase_label",
    "rename_tool",
    "reorder_schema_fields",
    "substitute_role_fillers",
    "swap_surface_domain",
]
