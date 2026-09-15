"""External V5 coordinate identities and append-only failure, repair, regression lineage."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from ..epistemic_cases import (
    FALLBACK_KEY,
    contract_provenance,
    evaluation_identity,
    get_case,
    is_canonical_case,
    resolve_legacy_case_name,
    stripe_registry,
    validate_coordinate,
)
from .entities import OntologyEntity, Provenance
from .graph import OntologyGraph, OntologyRelation


def _validate_supplied_identity(payload: Mapping[str, Any], identity, contract) -> None:
    """Reject supplied V5 identity fields that conflict with the resolved contract."""
    for name in (
        "framework_version",
        "canonical_case_id",
        "canonical_case_key",
        "canonical_case_title",
        "canonical",
    ):
        if name in payload:
            value = payload[name]
            if type(value) is not type(identity[name]) or value != identity[name]:
                raise ValueError(f"Supplied {name} contradicts the pinned V5 identity")
    if "contract_provenance" in payload and payload["contract_provenance"] != contract:
        raise ValueError("Supplied contract provenance differs from the pinned V5 mirror")
    if "base_case_name" in payload:
        expected = identity["canonical_case_key"] or FALLBACK_KEY
        if resolve_legacy_case_name(payload["base_case_name"]) != expected:
            raise ValueError("Supplied base_case_name contradicts the case_id")


def adapt_evaluation(
    record: Any, evaluation_id: str, provenance: tuple[Provenance, ...]
) -> OntologyGraph:
    """Resolve V5 through the shared contract; Case 0 has no canonical case entity or edge."""
    payload = dict(record) if isinstance(record, Mapping) else record.to_dict()
    case_id = payload["case_id"]
    stripe = payload.get("stripe", "NONE")
    subtype = payload.get("stripe_subtype")
    repeat = payload.get("repeat_id", 0)
    validate_coordinate(case_id, stripe, subtype, repeat)
    contract = contract_provenance()
    _validate_supplied_identity(
        payload, evaluation_identity(case_id, stripe, subtype, repeat), contract
    )
    namespace = contract["framework_version"]
    graph = OntologyGraph("evaluation", metadata={"contract": contract})
    canonical = is_canonical_case(case_id)
    graph.add_entity(
        OntologyEntity(
            evaluation_id,
            "EVALUATION_RECORD",
            evaluation_id,
            provenance=provenance,
            metadata={
                "source_payload": payload,
                "case_id": case_id,
                "canonical": canonical,
                "stripe": stripe,
                "stripe_subtype": subtype,
                "repeat_id": repeat,
                "contract": contract,
            },
        )
    )
    contract_source = (
        Provenance(
            contract["upstream_repository"],
            version=contract["upstream_commit"],
            created_by="pinned_contract_adapter",
        ),
    )
    if canonical:
        case = get_case(case_id)
        identifier = f"{namespace}:case:{case.id}"
        graph.add_entity(
            OntologyEntity(
                identifier,
                "CANONICAL_CASE",
                case.title,
                metadata={"case_id": case.id, "contract_case": asdict(case), "contract": contract},
                provenance=contract_source,
            )
        )
        graph.add_relation(OntologyRelation(evaluation_id, "HAS_CASE", identifier))
    stripe_id = f"{namespace}:stripe:{stripe}"
    stripe_record = stripe_registry()[stripe]
    graph.add_entity(
        OntologyEntity(
            stripe_id,
            "ROBUSTNESS_STRIPE",
            stripe_record["title"],
            metadata={"stripe_id": stripe, "contract_stripe": stripe_record, "contract": contract},
            provenance=contract_source,
        )
    )
    graph.add_relation(OntologyRelation(evaluation_id, "HAS_STRIPE", stripe_id))
    if subtype is not None:
        subtype_id = f"{namespace}:subtype:{stripe}:{subtype}"
        graph.add_entity(
            OntologyEntity(
                subtype_id,
                "STRIPE_SUBTYPE",
                subtype,
                metadata={"stripe_id": stripe, "subtype": subtype, "contract": contract},
                provenance=contract_source,
            )
        )
        graph.add_relation(OntologyRelation(evaluation_id, "HAS_SUBTYPE", subtype_id))
    repeat_id = f"{evaluation_id}:repeat:{repeat}"
    graph.add_entity(
        OntologyEntity(
            repeat_id,
            "REPEAT",
            str(repeat),
            metadata={"repeat_id": repeat},
            provenance=provenance,
        )
    )
    graph.add_relation(OntologyRelation(evaluation_id, "HAS_REPEAT", repeat_id))
    return graph


def add_failure_repair(
    graph: OntologyGraph,
    failure: OntologyEntity,
    failed_evaluation_id: str,
    repair: OntologyEntity,
    verification_evaluation_id: str,
    regression: OntologyEntity,
) -> None:
    """Append a repair attempt and its retest link; the retest can record success or failure.

    VERIFIED_BY identifies the evaluation that checks the repair, not its result.
    """
    # Validate on a temporary graph so a malformed lineage cannot partially mutate history.
    proposed = OntologyGraph(graph.view, metadata=graph.metadata)
    for entity in graph.entities.values():
        proposed.add_entity(entity)
    for entity in (failure, repair, regression):
        proposed.add_entity(entity)
    edges = (
        OntologyRelation(failure.id, "OBSERVED_IN", failed_evaluation_id),
        OntologyRelation(failure.id, "REPAIRED_BY", repair.id),
        OntologyRelation(repair.id, "VERIFIED_BY", verification_evaluation_id),
        OntologyRelation(regression.id, "REGRESSION_OF", failure.id),
        OntologyRelation(regression.id, "VERIFIES", repair.id),
        OntologyRelation(verification_evaluation_id, "PRODUCES", regression.id),
    )
    for edge in (*graph.relations, *edges):
        proposed.add_relation(edge)
    proposed.validate()
    for entity in (failure, repair, regression):
        graph.add_entity(entity)
    for edge in edges:
        graph.add_relation(edge)
