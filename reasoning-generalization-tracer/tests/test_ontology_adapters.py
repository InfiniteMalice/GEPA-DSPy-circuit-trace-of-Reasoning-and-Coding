"""Existing graph formats remain intact when connected to ontology views."""

import pytest

from rg_tracer.attribution.schema import AttributionGraph
from rg_tracer.concept_lattices.types import ConceptAttribute, ConceptLatticeSpec
from rg_tracer.epistemic_cases import get_case_title, stripe_registry
from rg_tracer.ontology import OntologyEntity, OntologyRelation, Provenance
from rg_tracer.ontology.adapters import (
    adapt_attribution_graph,
    adapt_concept_lattice,
    adapt_reasoning_graph,
    adapt_reasoning_units,
    adapt_semantic_tag,
    adapt_transformation,
)
from rg_tracer.ontology.evaluation import adapt_evaluation, add_failure_repair
from rg_tracer.reasoning_graphs.schema import ReasoningGraph
from rg_tracer.schema_v3.reasoning_units import REASONING_UNIT_REGISTRY
from rg_tracer.semantics.taxonomy import SemanticTag


def test_reasoning_adapter_preserves_source_identity_metadata_and_unknown_relations():
    """Verify that reasoning adapter preserves source identity metadata and unknown relations."""
    payload = {
        "task_id": "task",
        "answer_ref": "answer",
        "metadata": {"run": 3},
        "nodes": [
            {
                "id": "c",
                "label": "claim",
                "kind": "claim",
                "text": "x is 2",
                "metadata": {"confidence": 0.9},
            },
            {
                "id": "e",
                "label": "check",
                "kind": "evidence",
                "text": "test output",
                "metadata": {"source": "checker"},
            },
        ],
        "edges": [
            {
                "source": "e",
                "target": "c",
                "relation": "supports",
                "weight": 0.4,
                "metadata": {"source_span": "line 5"},
            },
            {
                "source": "c",
                "target": "e",
                "relation": "unmapped_custom",
                "metadata": {"preserve": True},
            },
        ],
    }
    source = ReasoningGraph.from_mapping(payload)
    result = adapt_reasoning_graph(source)
    assert result.source_payload == source.to_dict()
    assert set(result.graph.entities) == {"c", "e"}
    assert result.graph.entities["c"].epistemic_status.value == "INFERRED"
    assert result.graph.relations[0].metadata["original_relation"] == "supports"
    assert result.graph.relations[0].metadata["source_edge"]["weight"] == 0.4
    assert len(result.unmapped) == 1
    assert source.to_dict() == payload
    result.graph.validate()


def test_lattice_adapter_preserves_shadow_semantics_and_implication():
    """Verify that lattice adapter preserves shadow semantics and implication."""
    spec = ConceptLatticeSpec(
        "shape",
        "geometry",
        (
            ConceptAttribute("square", "four equal sides"),
            ConceptAttribute("rectangle", "four right angles"),
        ),
        (("square", "rectangle"),),
        shadow_only=True,
    )
    result = adapt_concept_lattice(spec)
    assert result.graph.metadata["shadow_only"] is True
    assert result.graph.relations[0].relation.value == "ENTAILS"
    assert all(value.type.value == "CONCEPT" for value in result.graph.entities.values())
    assert result.source_payload["attributes"][0]["description"] == "four equal sides"
    assert result.graph.relations[0].metadata["epistemic_status"] == "PROVISIONAL"


def test_attribution_adapter_preserves_mechanistic_provenance_and_extra_fields():
    """Verify that attribution adapter preserves mechanistic provenance and extra fields."""
    payload = {
        "model_ref": "model",
        "task_id": "task",
        "meta": {
            "token_positions": [1, 3],
            "phase": "answer",
            "extra": "keep",
        },
        "nodes": [
            {"id": "f", "layer": 2, "type": "feature", "activation": 0.7},
            {"id": "g", "layer": 3, "type": "circuit", "activation": 0.9},
        ],
        "edges": [{"src": "f", "dst": "g", "attr": -0.3}],
    }
    source = AttributionGraph.from_mapping(payload)
    result = adapt_attribution_graph(source)
    assert result.source_payload == source.to_dict()
    feature = result.graph.entities["f"]
    assert feature.metadata["layer"] == 2
    assert feature.metadata["activation"] == 0.7
    assert feature.metadata["model_ref"] == "model"
    assert feature.metadata["task_id"] == "task"
    assert feature.metadata["token_positions"] == (1, 3)
    assert feature.metadata["phase"] == "answer"
    assert result.graph.relations[0].metadata["attr"] == -0.3
    assert result.graph.relations[0].relation.value == "CONTRIBUTES_TO"
    payload["nodes"][0]["token_position"] = 8
    assert adapt_attribution_graph(payload).graph.entities["f"].metadata["token_position"] == 8


def test_semantic_tag_adapter_reuses_taxonomy_and_does_not_call_success_a_failure():
    """Verify that semantic tag adapter reuses taxonomy and does not call success a failure."""
    for tag in SemanticTag:
        entity = adapt_semantic_tag(tag)
        assert entity.canonical_name == tag.value
        assert entity.id == f"semantic-tag:{tag.value}"
        expected = (
            "CONCEPT" if tag in {SemanticTag.SUPPORTED, SemanticTag.ENTAILED} else "FAILURE_FAMILY"
        )
        assert entity.type.value == expected
    with pytest.raises(ValueError):
        adapt_semantic_tag("invented")


def test_reasoning_registry_integrates_families_dependencies_and_unknown_partners():
    """Verify that reasoning registry integrates families dependencies and unknown partners."""
    result = adapt_reasoning_units()
    for name, source in REASONING_UNIT_REGISTRY.items():
        assert result.graph.entities[f"reasoning-family:{name}"].type.value == "CONCEPT"
        unit = result.graph.entities[f"reasoning-unit:{name}"]
        assert unit.metadata["registry_entry"]["definition"] == source.definition
    assert any(edge.relation.value == "DEPENDS_ON" for edge in result.graph.relations)
    assert any("scientific_method_check" in item["reason"] for item in result.unmapped)


def test_v5_adapter_uses_shared_identity_and_case_zero_stays_operational():
    """Verify that v5 adapter uses shared identity and case zero stays operational."""
    provenance = (Provenance("evaluation-log", run_id="run"),)
    graph = adapt_evaluation({"case_id": 6, "stripe": "NONE", "repeat_id": 2}, "ev", provenance)
    assert graph.entities["17case-v5:case:6"].canonical_name == get_case_title(6)
    assert {r.relation.value for r in graph.relations} == {"HAS_CASE", "HAS_STRIPE", "HAS_REPEAT"}
    fallback = adapt_evaluation({"case_id": 0}, "fallback", provenance)
    assert fallback.entities["fallback"].metadata["canonical"] is False
    assert not any(value.type.value == "CANONICAL_CASE" for value in fallback.entities.values())
    assert not any(edge.relation.value == "HAS_CASE" for edge in fallback.relations)
    with pytest.raises(ValueError):
        adapt_evaluation({"case_id": 18}, "bad", provenance)
    with pytest.raises(ValueError):
        OntologyEntity("17case-v5:case:0", "CANONICAL_CASE", "fallback", metadata={"case_id": 0})


def test_v5_subtype_is_resolved_in_shared_stripe_namespace():
    """Verify that v5 subtype is resolved in shared stripe namespace."""
    stripe, row = next(
        (key, row) for key, row in stripe_registry().items() if row["allowed_subtypes"]
    )
    subtype = row["allowed_subtypes"][0]
    graph = adapt_evaluation(
        {"case_id": 6, "stripe": stripe, "stripe_subtype": subtype},
        "ev",
        (Provenance("eval"),),
    )
    assert graph.entities[f"17case-v5:subtype:{stripe}:{subtype}"].canonical_name == subtype
    graph.validate()


def test_failure_repair_adds_history_and_regression_without_overwriting_failure():
    """Verify that failure repair adds history and regression without overwriting failure."""
    p = (Provenance("evaluation-log"),)
    graph = adapt_evaluation({"case_id": 6}, "failed", p)
    verification = OntologyEntity("retest", "EVALUATION_RECORD", "retest", provenance=p)
    graph.add_entity(verification)
    failure = OntologyEntity("failure", "FAILURE", "unsupported", provenance=p)
    repair = OntologyEntity("repair", "REPAIR", "add independent check", provenance=p)
    regression = OntologyEntity("regression", "REGRESSION_TEST", "check unsupported", provenance=p)
    add_failure_repair(graph, failure, "failed", repair, "retest", regression)
    assert graph.entities["failure"] is failure
    assert {r.relation.value for r in graph.relations} >= {
        "OBSERVED_IN",
        "REPAIRED_BY",
        "VERIFIED_BY",
        "VERIFIES",
        "REGRESSION_OF",
    }
    graph.validate()


def test_group_overlay_transformation_records_symmetry_evidence_without_proving_equivalence():
    """Verify group overlays record symmetry evidence without proving equivalence."""
    value = adapt_transformation(
        {
            "name": "reframing",
            "kind": "PromptReframing",
            "expected_invariants": ["authorization"],
            "observed_invariants": [],
            "symmetry_breaks": ["authorization"],
            "group_overlay": {"same_equivalence_class": False},
        },
        "t",
        (Provenance("group-overlay"),),
    )
    assert value.symmetry_breaks == ("authorization",)
    assert value.metadata["source_payload"]["group_overlay"]["same_equivalence_class"] is False


def test_direct_stripe_entities_cannot_redefine_contract_identity():
    """Verify that direct stripe entities cannot redefine contract identity."""
    with pytest.raises(ValueError):
        OntologyEntity("invented", "ROBUSTNESS_STRIPE", "invented", metadata={"stripe_id": "NONE"})
    with pytest.raises(ValueError):
        OntologyEntity(
            "invented",
            "STRIPE_SUBTYPE",
            "invented",
            metadata={
                "stripe_id": "NONE",
                "subtype": "invented",
            },
        )


def test_case_zero_cannot_acquire_a_canonical_case_edge():
    """Verify that case zero cannot acquire a canonical case edge."""
    p = (Provenance("eval"),)
    fallback = adapt_evaluation({"case_id": 0}, "fallback", p)
    canonical = adapt_evaluation({"case_id": 1}, "canonical", p)
    fallback.add_entity(canonical.entities["17case-v5:case:1"])
    with pytest.raises(ValueError):
        fallback.add_relation(OntologyRelation("fallback", "HAS_CASE", "17case-v5:case:1"))


def test_failure_repair_is_atomic_when_retest_is_missing():
    """Verify that failure repair is atomic when retest is missing."""
    p = (Provenance("eval"),)
    graph = adapt_evaluation({"case_id": 6}, "failed", p)
    before = graph.to_dict()
    values = [
        OntologyEntity(key, kind, key, provenance=p)
        for key, kind in (
            ("f", "FAILURE"),
            ("r", "REPAIR"),
            ("t", "REGRESSION_TEST"),
        )
    ]
    with pytest.raises(ValueError):
        add_failure_repair(graph, values[0], "failed", values[1], "missing", values[2])
    assert graph.to_dict() == before


@pytest.mark.parametrize(
    "extra",
    [
        {"framework_version": "17case-v4"},
        {"canonical": False},
        {"canonical": 1},
        {"canonical_case_id": 2},
        {"canonical_case_id": True},
        {"canonical_case_key": "wrong"},
        {"canonical_case_title": "wrong"},
        {"contract_provenance": {"upstream_commit": "wrong"}},
        {"contract_provenance": None},
        {"base_case_name": "lazy_sandbagging_idk"},
    ],
)
def test_v5_adapter_rejects_conflicting_source_identity(extra):
    """Verify that v5 adapter rejects conflicting source identity."""
    with pytest.raises(ValueError):
        adapt_evaluation({"case_id": 1, **extra}, "evaluation", (Provenance("log"),))


def test_v5_adapter_accepts_valid_metadata_and_explicit_legacy_absence():
    """Verify that v5 adapter accepts valid metadata and explicit legacy absence."""
    from rg_tracer.epistemic_cases import contract_provenance, evaluation_identity

    payload = {**evaluation_identity(1), "contract_provenance": contract_provenance()}
    graph = adapt_evaluation(payload, "ev", (Provenance("log"),))
    assert graph.entities["ev"].metadata["source_payload"]["canonical_case_id"] == 1
    graph = adapt_evaluation({"case_id": 1}, "legacy", (Provenance("legacy"),))
    assert graph.entities["legacy"].metadata["canonical"] is True


def test_reasoning_adapter_contradiction_blocks_task_verification():
    """Verify that reasoning adapter contradiction blocks task verification."""
    payload = {
        "nodes": [
            {"id": "claim", "kind": "claim", "label": "claim", "text": "x is 2"},
            {"id": "support", "kind": "evidence", "label": "support", "text": "x = 2"},
            {"id": "counter", "kind": "evidence", "label": "counter", "text": "x = 3"},
        ],
        "edges": [
            {"source": "support", "target": "claim", "relation": "supports"},
            {"source": "counter", "target": "claim", "relation": "contradicts"},
        ],
    }
    result = adapt_reasoning_graph(payload)
    graph = result.graph
    value = graph.entities["claim"]
    assert value.supporting_evidence == ("support",)
    assert value.contradicting_evidence == ("counter",)
    assert value.epistemic_status.value == "CONTRADICTED"
    assert result.source_payload == payload
    graph.add_entity(
        OntologyEntity(
            "check",
            "VERIFICATION",
            "check",
            provenance=(Provenance("task-check"),),
            metadata={"method": "task_verification", "result": "passed", "claim_id": "claim"},
        )
    )
    with pytest.raises(ValueError, match="disputed"):
        graph.promote_claim("claim", "check")
    graph.validate()


@pytest.mark.parametrize(
    "adapter, field_name, expected_type",
    [
        (adapt_reasoning_graph, "kind", "REPRESENTATION"),
        (adapt_attribution_graph, "type", "MODEL_FEATURE"),
    ],
)
def test_raw_adapter_nullable_node_type_uses_fallback_and_preserves_source(
    adapter, field_name, expected_type
):
    """Verify that raw adapter nullable node type uses fallback and preserves source."""
    payload = {"nodes": [{"id": "node", field_name: None}], "edges": []}
    result = adapter(payload)
    assert result.graph.entities["node"].type.value == expected_type
    assert result.source_payload == payload
    assert result.source_payload["nodes"][0][field_name] is None
    result.graph.validate()


@pytest.mark.parametrize(
    "adapter, endpoints",
    [
        (adapt_reasoning_graph, {"source": "a", "target": "b"}),
        (adapt_attribution_graph, {"src": "a", "dst": "b"}),
    ],
)
def test_raw_adapter_nullable_relation_remains_unmapped_and_preserves_source(adapter, endpoints):
    """Verify that raw adapter nullable relation remains unmapped and preserves source."""
    payload = {
        "nodes": [
            {"id": "a", "kind": "concept", "type": "feature"},
            {"id": "b", "kind": "concept", "type": "feature"},
        ],
        "edges": [{**endpoints, "relation": None}],
    }
    result = adapter(payload)
    assert result.graph.relations == ()
    assert len(result.unmapped) == 1
    assert result.source_payload == payload
    assert result.source_payload["edges"][0]["relation"] is None
    result.graph.validate()
