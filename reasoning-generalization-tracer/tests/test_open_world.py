import json

from rg_tracer.open_world import (
    add_optional_field,
    add_stale_observation,
    introduce_ambiguity,
    swap_surface_domain,
)


def test_perturbations_preserve_task_id_lineage():
    task = {"task_id": "t1", "prompt": "Solve the biology puzzle"}
    shifted = swap_surface_domain(task, "biology", "chemistry", seed=7)
    assert shifted.parent_task_id == "t1"
    assert shifted.task_id.startswith("t1:domain_surface_swap")


def test_perturbations_are_deterministic_with_seed():
    task = {"task_id": "t1", "prompt": "Solve"}
    first = introduce_ambiguity(task, seed=11)
    second = introduce_ambiguity(task, seed=11)
    assert first.prompt == second.prompt
    assert first.metadata == second.metadata
    assert first.task_id != second.task_id


def test_high_stakes_ambiguity_flag_is_set():
    task = {"task_id": "t1", "prompt": "Prescribe a treatment"}
    shifted = introduce_ambiguity(task, high_stakes=True)
    assert shifted.metadata["high_stakes_ambiguity"] is True


def test_schema_drift_metadata_is_present():
    schema = {"properties": {"b": {"type": "string"}}}
    shifted = add_optional_field(schema, "a", {"type": "integer"})
    assert shifted["metadata"]["schema_drift"] is True


def test_generated_records_remain_json_serializable():
    observation = {"value": 1}
    shifted = add_stale_observation(observation, "old")
    assert json.loads(json.dumps(shifted))["stale_observation"] == "old"
