"""Lossless source snapshots plus conservative typed projections of specialized graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .entities import Claim, OntologyEntity, Provenance, Transformation, serialize
from .graph import OntologyGraph, OntologyRelation


@dataclass(frozen=True)
class AdapterResult:
    """Projection diagnostics never erase source nodes, edges, or unknown metadata."""

    graph: OntologyGraph
    unmapped: tuple[dict[str, Any], ...] = ()

    @property
    def source_payload(self) -> dict[str, Any]:
        """Return a defensive JSON-compatible snapshot, including unmapped source data."""
        return serialize(self.graph.metadata["source_payload"])


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value.as_dict()


def _edge(graph, source, relation, target, original, unmapped):
    try:
        graph.add_relation(OntologyRelation(source, relation, target, metadata=original))
    except ValueError as error:
        unmapped.append({"source_edge": original, "reason": str(error)})


def adapt_reasoning_graph(source: Any, *, source_ref="ReasoningGraph") -> AdapterResult:
    """Project public reasoning; claims start inferred and contradictory edges mark disputes."""
    payload = _payload(source)
    graph = OntologyGraph("semantic", metadata={"source_payload": payload})
    unmapped = []
    kinds = {
        "claim": "CLAIM",
        "conclusion": "CLAIM",
        "premise": "CLAIM",
        "evidence": "EVIDENCE",
        "concept": "CONCEPT",
        "action": "ACTION",
        "tool_action": "TOOL_ACTION",
        "reasoning_unit": "REASONING_UNIT",
        "step": "REASONING_UNIT",
        "assumption": "ASSUMPTION",
        "constraint": "CONSTRAINT",
    }
    for node in payload.get("nodes", []):
        kind = kinds.get((node.get("kind") or "").lower(), "REPRESENTATION")
        provenance = (
            Provenance(
                source=source_ref,
                source_span=node["id"],
                run_id=payload.get("metadata", {}).get("run_id"),
                step_id=node["id"],
                created_by="reasoning_graph_adapter",
            ),
        )
        fields = {
            "id": node["id"],
            "canonical_name": node.get("label") or node["id"],
            "metadata": {
                "source_node": node,
                "task_id": payload.get("task_id"),
                "answer_ref": payload.get("answer_ref"),
            },
            "provenance": provenance,
        }
        if kind == "CLAIM":
            value = Claim(**fields, content=node.get("text"))
        else:
            value = OntologyEntity(**fields, type=kind)
        graph.add_entity(value)
        if kind == "REPRESENTATION":
            unmapped.append(
                {"node_id": node["id"], "reason": "Unknown node kind retained as representation"}
            )
    for edge in payload.get("edges", []):
        relation = (edge.get("relation") or "").upper()
        _edge(
            graph,
            edge["source"],
            relation,
            edge["target"],
            {
                "source_edge": edge,
                "original_relation": edge.get("relation"),
            },
            unmapped,
        )
    graph.validate()
    return AdapterResult(graph, tuple(unmapped))


def adapt_concept_lattice(source: Any, *, source_ref="ConceptLatticeSpec") -> AdapterResult:
    """Project diagnostic attribute implications without promoting them to world knowledge."""
    payload = _payload(source)
    graph = OntologyGraph(
        "semantic",
        metadata={
            "source_payload": payload,
            "shadow_only": payload.get("shadow_only", True),
            "domain": payload["domain"],
            "epistemic_status": "PROVISIONAL",
        },
    )
    ids = {}
    for attribute in payload["attributes"]:
        identifier = f"lattice:{payload['domain']}:{payload['name']}:{attribute['name']}"
        ids[attribute["name"]] = identifier
        graph.add_entity(
            OntologyEntity(
                identifier,
                "CONCEPT",
                attribute["name"],
                metadata={"attribute": attribute, "diagnostic": True},
                provenance=(Provenance(source_ref, source_span=attribute["name"]),),
            )
        )
    unmapped = []
    for source_name, target_name in payload.get("implications", []):
        _edge(
            graph,
            ids.get(source_name, source_name),
            "ENTAILS",
            ids.get(target_name, target_name),
            {
                "implication": [source_name, target_name],
                "epistemic_status": "PROVISIONAL",
                "shadow_only": payload.get("shadow_only", True),
            },
            unmapped,
        )
    return AdapterResult(graph, tuple(unmapped))


def adapt_attribution_graph(source: Any, *, source_ref="AttributionGraph") -> AdapterResult:
    """Preserve mechanistic observations; attribution magnitude does not imply causality."""
    payload = _payload(source)
    graph = OntologyGraph("mechanistic", metadata={"source_payload": payload})
    meta = payload.get("meta", {})
    context = {
        "model_ref": payload.get("model_ref"),
        "task_id": payload.get("task_id"),
        "token_positions": meta.get("token_positions", []),
        "phase": meta.get("phase"),
    }
    for node in payload.get("nodes", []):
        kind = (
            "CIRCUIT_FEATURE" if "circuit" in (node.get("type") or "").lower() else "MODEL_FEATURE"
        )
        graph.add_entity(
            OntologyEntity(
                node["id"],
                kind,
                node["id"],
                metadata={**context, **node},
                provenance=(
                    Provenance(
                        source_ref,
                        source_span=node["id"],
                        run_id=meta.get("run_id"),
                        created_by="attribution_graph_adapter",
                        version=payload.get("model_ref"),
                    ),
                ),
            )
        )
    unmapped = []
    for edge in payload.get("edges", []):
        relation = (edge.get("relation", "CONTRIBUTES_TO") or "").upper()
        if relation not in {"CONTRIBUTES_TO", "ATTRIBUTED_TO"}:
            unmapped.append({"source_edge": edge, "reason": "Attribution is not a causal test"})
            continue
        _edge(graph, edge["src"], relation, edge["dst"], {**context, **edge}, unmapped)
    return AdapterResult(graph, tuple(unmapped))


def adapt_semantic_tag(tag: Any) -> OntologyEntity:
    """Reuse SemanticTag values; SUPPORTED and ENTAILED denote validation concepts."""
    from ..semantics.taxonomy import SemanticTag

    canonical = SemanticTag(tag)
    success = canonical in {SemanticTag.SUPPORTED, SemanticTag.ENTAILED}
    return OntologyEntity(
        f"semantic-tag:{canonical.value}",
        "CONCEPT" if success else "FAILURE_FAMILY",
        canonical.value,
        metadata={"taxonomy": "SemanticTag", "tag": canonical.value},
        provenance=(Provenance("rg_tracer.semantics.taxonomy.SemanticTag"),),
    )


def adapt_reasoning_units() -> AdapterResult:
    """Reuse registered families as concepts and connect their executable unit descriptions."""
    from ..schema_v3.reasoning_units import REASONING_UNIT_REGISTRY

    source = {name: entry.to_dict() for name, entry in REASONING_UNIT_REGISTRY.items()}
    graph = OntologyGraph("semantic", metadata={"source_payload": source})
    provenance = (Provenance("rg_tracer.schema_v3.reasoning_units.REASONING_UNIT_REGISTRY"),)
    for name, entry in source.items():
        concept_id, unit_id = f"reasoning-family:{name}", f"reasoning-unit:{name}"
        graph.add_entity(OntologyEntity(concept_id, "CONCEPT", name, provenance=provenance))
        graph.add_entity(
            OntologyEntity(
                unit_id,
                "REASONING_UNIT",
                name,
                metadata={"registry_entry": entry},
                provenance=provenance,
            )
        )
        graph.add_relation(OntologyRelation(unit_id, "INSTANCE_OF", concept_id))
    unmapped = []
    for name, entry in source.items():
        for field_name, relation in (
            ("dependencies", "DEPENDS_ON"),
            ("composition_partners", "COMPOSES_WITH"),
        ):
            for target in entry[field_name]:
                _edge(
                    graph,
                    f"reasoning-unit:{name}",
                    relation,
                    f"reasoning-unit:{target}",
                    {"registry_field": field_name},
                    unmapped,
                )
    return AdapterResult(graph, tuple(unmapped))


def adapt_transformation(
    payload: Mapping[str, Any], identifier: str, provenance: tuple[Provenance, ...]
) -> Transformation:
    """Accept group-overlay results as reported invariants, retaining the complete overlay."""
    names = (
        "inputs",
        "outputs",
        "preserved_properties",
        "changed_properties",
        "expected_invariants",
        "observed_invariants",
        "symmetry_breaks",
        "inverse",
        "composition_parent",
    )
    return Transformation(
        id=identifier,
        canonical_name=payload.get("name", identifier),
        kind=payload.get("kind", "RepresentationChange"),
        provenance=provenance,
        metadata={"source_payload": dict(payload)},
        **{name: payload[name] for name in names if name in payload},
    )
