"""Ontology contracts protect provenance, identity, and evidence boundaries."""

from dataclasses import replace

import pytest

from rg_tracer.ontology import (
    CanonicalIdentity,
    Claim,
    GraphView,
    OntologyEntity,
    OntologyGraph,
    OntologyRelation,
    Provenance,
    Transformation,
    candidate_equivalences,
    canonicalize_claims,
    load_registry,
)


def provenance(source="test", **kwargs):
    """Build test provenance for ``source``."""
    return (Provenance(source=source, **kwargs),)


def entity(identifier, kind, **kwargs):
    """Build a typed test entity with provenance."""
    return OntologyEntity(identifier, kind, identifier, provenance=provenance(), **kwargs)


def claim(identifier="claim", **kwargs):
    """Build a test claim with caller-supplied overrides."""
    return Claim(id=identifier, canonical_name=identifier, provenance=provenance(), **kwargs)


def test_registry_is_versioned_and_relation_semantics_are_complete():
    """Verify that registry is versioned and relation semantics are complete."""
    registry = load_registry()
    assert registry["ontology_version"] == "rg-ontology-v1"
    for relation in registry["relations"].values():
        assert {"signatures", "direction", "inverse", "transitive", "symmetric"} <= relation.keys()
    registry["entity_types"].clear()
    assert load_registry()["entity_types"]


def test_invalid_entities_and_missing_provenance_are_rejected():
    """Verify that invalid entities and missing provenance are rejected."""
    for identifier, kind in [("", "CONCEPT"), ("x", "INVENTED")]:
        with pytest.raises(ValueError):
            entity(identifier, kind)
    with pytest.raises(ValueError, match="provenance"):
        OntologyEntity("c", "CLAIM", "claim")
    with pytest.raises(ValueError, match="Claim"):
        entity("c", "CLAIM")
    with pytest.raises(ValueError, match="verification"):
        claim(epistemic_status="ESTABLISHED")
    with pytest.raises(ValueError):
        claim(confidence=float("nan"), confidence_source="model")


def test_semantic_cycles_allowed_but_mixed_provenance_cycle_rejected_atomically():
    """Verify that semantic cycles allowed but mixed provenance cycle rejected atomically."""
    semantic = OntologyGraph(GraphView.SEMANTIC)
    for identifier in ("a", "b"):
        semantic.add_entity(claim(identifier))
    semantic.add_relation(OntologyRelation("a", "DEPENDS_ON", "b"))
    semantic.add_relation(OntologyRelation("b", "DEPENDS_ON", "a"))
    semantic.validate()
    execution = OntologyGraph(GraphView.EXECUTION)
    execution.add_entity(entity("s", "STEP"))
    execution.add_entity(entity("a", "ARTIFACT"))
    execution.add_entity(entity("t", "STEP"))
    execution.add_relation(OntologyRelation("s", "PRECEDES", "t"))
    execution.add_relation(OntologyRelation("t", "DERIVED_FROM", "s"))
    # DERIVED_FROM points from derivative to source, so temporal order is reversed.
    with pytest.raises(ValueError, match="cycle"):
        execution.add_relation(OntologyRelation("t", "PRECEDES", "s"))
    assert len(execution.relations) == 2


def test_relation_domain_range_and_view_are_enforced():
    """Verify that relation domain range and view are enforced."""
    graph = OntologyGraph("semantic")
    graph.add_entity(claim())
    graph.add_entity(entity("concept", "CONCEPT"))
    with pytest.raises(ValueError):
        graph.add_relation(OntologyRelation("concept", "SUPPORTS", "claim"))
    with pytest.raises(ValueError):
        graph.add_relation(OntologyRelation("claim", "PROVES", "concept"))


def test_conflicting_evidence_and_status_history_are_non_destructive():
    """Verify that conflicting evidence and status history are non destructive."""
    graph = OntologyGraph("semantic")
    graph.add_entity(claim())
    graph.add_entity(entity("positive", "EVIDENCE"))
    graph.add_entity(entity("negative", "EVIDENCE"))
    graph.record_evidence("claim", "positive")
    graph.record_evidence("claim", "negative", contradicts=True)
    updated = graph.entities["claim"]
    assert updated.supporting_evidence == ("positive",)
    assert updated.contradicting_evidence == ("negative",)
    assert updated.epistemic_status.value == "CONTRADICTED"
    assert updated.status_history[0] == "INFERRED"
    assert len(graph.relations) == 2
    assert len(graph.entities) == 3


def test_mechanistic_observations_cannot_certify_claims_or_equate_features():
    """Verify that mechanistic observations cannot certify claims or equate features."""
    graph = OntologyGraph("mechanistic")
    graph.add_entity(entity("feature", "CIRCUIT_FEATURE"))
    graph.add_entity(entity("concept", "CONCEPT"))
    graph.add_entity(entity("observation", "MECHANISTIC_OBSERVATION"))
    graph.add_entity(claim())
    graph.add_relation(OntologyRelation("observation", "SUPPORTS_MAPPING", "concept"))
    for relation in ("EQUIVALENT_TO", "CAUSALLY_CONTRIBUTES_TO"):
        with pytest.raises(ValueError):
            graph.add_relation(OntologyRelation("feature", relation, "concept"))
    with pytest.raises(ValueError):
        graph.add_relation(OntologyRelation("observation", "SUPPORTS", "claim"))
    graph.add_entity(
        entity(
            "intervention",
            "VERIFICATION",
            metadata={
                "method": "intervention",
                "result": "passed",
                "source_id": "feature",
                "target_id": "concept",
            },
        )
    )
    graph.add_relation(
        OntologyRelation(
            "feature", "CAUSALLY_CONTRIBUTES_TO", "concept", evidence_ids=("intervention",)
        )
    )


def test_claim_promotion_requires_task_verification_and_support():
    """Verify that claim promotion requires task verification and support."""
    graph = OntologyGraph("semantic")
    graph.add_entity(claim())
    graph.add_entity(entity("observation", "MECHANISTIC_OBSERVATION"))
    with pytest.raises(ValueError):
        graph.promote_claim("claim", "observation")
    graph.add_entity(entity("evidence", "EVIDENCE"))
    graph.record_evidence("claim", "evidence")
    graph.add_entity(
        entity(
            "verification",
            "VERIFICATION",
            metadata={
                "method": "task_verification",
                "result": "passed",
                "claim_id": "claim",
            },
        )
    )
    graph.promote_claim("claim", "verification")
    assert graph.entities["claim"].epistemic_status.value == "VERIFIED"


def test_canonicalization_retains_observations_and_rejects_constraint_changes():
    """Verify that canonicalization retains observations and rejects constraint changes."""
    texts = ("parity remains unchanged", "the parity is invariant", "parity is preserved")
    claims = tuple(
        Claim(
            id=f"p{i}",
            canonical_name=text,
            semantic_key="parity_invariance",
            constraints={"domain": "integers"},
            provenance=provenance(f"source-{i}"),
        )
        for i, text in enumerate(texts)
    )
    assert len(candidate_equivalences(claims)) == 3
    verification = entity(
        "equivalence-check",
        "VERIFICATION",
        metadata={
            "method": "semantic_equivalence",
            "result": "passed",
            "observation_ids": [c.id for c in claims],
        },
    )
    canonical = canonicalize_claims(claims, "canonical:parity", verification)
    assert canonical.observations == claims
    assert len(canonical.provenance) == 3
    changed = replace(claims[1], constraints={"domain": "floats"})
    assert not candidate_equivalences((claims[0], changed))
    with pytest.raises(ValueError, match="constraints"):
        canonicalize_claims((claims[0], changed), "wrong", verification)
    assert not candidate_equivalences((claim("a"), claim("b")))


def test_authorization_difference_prevents_surface_equivalence():
    """Verify that authorization difference prevents surface equivalence."""
    defensive = claim("defensive", semantic_key="network_action", constraints={"authorized": True})
    harmful = claim("harmful", semantic_key="network_action", constraints={"authorized": False})
    assert candidate_equivalences((defensive, harmful)) == ()


def test_transformations_record_invariants_symmetry_breaks_and_composition():
    """Verify that transformations record invariants symmetry breaks and composition."""
    transformation = Transformation(
        id="rename",
        canonical_name="Variable rename",
        kind="VariableRename",
        inputs=("before",),
        outputs=("after",),
        expected_invariants=("binding",),
        observed_invariants=("binding",),
        symmetry_breaks=("printed_name",),
        preserved_properties=("meaning",),
        changed_properties=("syntax",),
        inverse="undo",
        composition_parent="pipeline",
        provenance=provenance(),
    )
    assert transformation.to_dict()["expected_invariants"] == ["binding"]
    assert transformation.to_dict()["inverse"] == "undo"


def test_stable_identity_and_nested_metadata_are_immutable():
    """Verify that stable identity and nested metadata are immutable."""
    original = {"nested": [1]}
    value = entity("x", "CONCEPT", metadata=original)
    original["nested"].append(2)
    assert value.to_dict()["metadata"]["nested"] == [1]
    graph = OntologyGraph("semantic")
    graph.add_entity(value)
    with pytest.raises(ValueError, match="identity"):
        graph.add_entity(replace(value, canonical_name="different"))


def test_strong_claim_cannot_bypass_verification_by_direct_construction():
    """Verify that strong claim cannot bypass verification by direct construction."""
    graph = OntologyGraph("semantic")
    graph.add_entity(claim(epistemic_status="VERIFIED", verification_ids=("missing",)))
    with pytest.raises(ValueError):
        graph.to_dict()


def test_causal_test_for_different_endpoints_does_not_authorize_edge():
    """Verify that causal test for different endpoints does not authorize edge."""
    graph = OntologyGraph("mechanistic")
    for identifier, kind in (("f", "MODEL_FEATURE"), ("c", "CONCEPT")):
        graph.add_entity(entity(identifier, kind))
    graph.add_entity(
        entity(
            "test",
            "VERIFICATION",
            metadata={
                "method": "intervention",
                "result": "passed",
                "source_id": "other",
                "target_id": "c",
            },
        )
    )
    with pytest.raises(ValueError, match="endpoint-bound"):
        graph.add_relation(
            OntologyRelation("f", "CAUSALLY_CONTRIBUTES_TO", "c", evidence_ids=("test",))
        )


def test_canonicalization_retains_conflicting_epistemic_observations():
    """Verify that canonicalization retains conflicting epistemic observations."""
    first = claim("a", semantic_key="parity")
    second = claim("b", semantic_key="parity", epistemic_status="CONTRADICTED")
    test = entity(
        "v",
        "VERIFICATION",
        metadata={
            "method": "semantic_equivalence",
            "result": "passed",
            "observation_ids": ["a", "b"],
        },
    )
    merged = canonicalize_claims((first, second), "canonical", test)
    assert merged.observations[0].epistemic_status.value == "INFERRED"
    assert merged.observations[1].epistemic_status.value == "CONTRADICTED"


def test_relation_definition_cannot_mutate_registry_boundary():
    """Verify that relation definition cannot mutate registry boundary."""
    from rg_tracer.ontology.registry import relation_definition

    definition = relation_definition("CAUSALLY_CONTRIBUTES_TO")
    definition["requires_intervention"] = False
    definition["signatures"][0]["range"].append("CLAIM")
    original = relation_definition("CAUSALLY_CONTRIBUTES_TO")
    assert original["requires_intervention"] is True
    assert "CLAIM" not in original["signatures"][0]["range"]


def supported_graph():
    """Build a graph containing a claim with direct supporting evidence."""
    graph = OntologyGraph("semantic")
    graph.add_entity(claim())
    graph.add_entity(entity("support", "EVIDENCE"))
    graph.add_entity(entity("counter", "EVIDENCE"))
    graph.add_entity(
        entity(
            "check",
            "VERIFICATION",
            metadata={
                "method": "task_verification",
                "result": "passed",
                "claim_id": "claim",
            },
        )
    )
    graph.add_relation(OntologyRelation("support", "SUPPORTS", "claim"))
    return graph


def test_direct_support_edge_enables_promotion_without_duplicate_bookkeeping():
    """Verify that direct support edge enables promotion without duplicate bookkeeping."""
    graph = supported_graph()
    assert graph.entities["claim"].supporting_evidence == ("support",)
    graph.record_evidence("claim", "support")
    assert graph.entities["claim"].supporting_evidence == ("support",)
    graph.promote_claim("claim", "check")
    assert graph.entities["claim"].epistemic_status.value == "VERIFIED"
    graph.validate()


def test_direct_contradiction_before_promotion_blocks_promotion_atomically():
    """Verify that direct contradiction before promotion blocks promotion atomically."""
    graph = supported_graph()
    original = graph.entities["claim"]
    graph.add_relation(OntologyRelation("counter", "CONTRADICTS", "claim"))
    current = graph.entities["claim"]
    assert current.contradicting_evidence == ("counter",)
    assert current.supporting_evidence == ("support",)
    assert current.provenance == original.provenance
    assert current.epistemic_status.value == "CONTRADICTED"
    assert current.status_history == ("INFERRED",)
    before = graph.to_dict()
    with pytest.raises(ValueError, match="disputed"):
        graph.promote_claim("claim", "check")
    assert graph.to_dict() == before


def test_direct_contradiction_after_promotion_preserves_verification_history():
    """Verify that direct contradiction after promotion preserves verification history."""
    graph = supported_graph()
    graph.record_evidence("claim", "support")
    graph.promote_claim("claim", "check")
    graph.add_relation(OntologyRelation("counter", "CONTRADICTS", "claim"))
    current = graph.entities["claim"]
    assert current.epistemic_status.value == "CONTRADICTED"
    assert current.status_history == ("INFERRED", "VERIFIED")
    assert current.verification_ids == ("check",)
    assert current.contradicting_evidence == ("counter",)
    assert len(graph.to_dict()["relations"]) == 4
    with pytest.raises(ValueError, match="disputed"):
        graph.promote_claim("claim", "check")


@pytest.mark.parametrize("reverse", [False, True])
def test_claim_contradiction_disputes_both_claims_without_using_claim_as_evidence(reverse):
    """Verify that claim contradiction disputes both claims without using claim as evidence."""
    graph = supported_graph()
    graph.record_evidence("claim", "support")
    graph.promote_claim("claim", "check")
    graph.add_entity(claim("other"))
    source, target = ("claim", "other") if reverse else ("other", "claim")
    graph.add_relation(OntologyRelation(source, "CONTRADICTS", target))
    for identifier in ("claim", "other"):
        value = graph.entities[identifier]
        assert value.epistemic_status.value == "CONTRADICTED"
        assert value.contradicting_evidence == ()
    with pytest.raises(ValueError, match="disputed"):
        graph.promote_claim("claim", "check")
    graph.validate()


def test_export_rejects_stale_verified_status_when_contradiction_edge_exists():
    """Verify that export rejects stale verified status when contradiction edge exists."""
    graph = supported_graph()
    graph.record_evidence("claim", "support")
    graph.promote_claim("claim", "check")
    verified = graph.entities["claim"]
    graph.add_entity(claim("other"))
    graph.add_relation(OntologyRelation("other", "CONTRADICTS", "claim"))
    # Simulate a stale reconstructed claim projection to exercise the independent export gate.
    graph._entities["claim"] = verified
    with pytest.raises(ValueError, match="undisputed"):
        graph.validate()
    with pytest.raises(ValueError, match="undisputed"):
        graph.to_dict()


def test_malformed_contradiction_does_not_modify_claim_or_relations():
    """Verify that malformed contradiction does not modify claim or relations."""
    graph = supported_graph()
    before = graph.to_dict()
    with pytest.raises(ValueError):
        graph.add_relation(
            OntologyRelation("counter", "CONTRADICTS", "claim", evidence_ids=("missing",))
        )
    assert graph.to_dict() == before


@pytest.mark.parametrize(
    "invalid",
    [
        "empty_id",
        "single_observation",
        "duplicate_ids",
        "different_semantic_key",
        "different_constraints",
        "wrong_declared_key",
        "unbound_verification",
        "failed_verification",
    ],
)
def test_direct_canonical_identity_constructor_enforces_factory_invariants(invalid):
    """Verify that direct canonical identity constructor enforces factory invariants."""
    first = claim("a", semantic_key="parity", constraints={"authorized": True})
    second = claim("b", semantic_key="parity", constraints={"authorized": True})
    observations = (first, second)
    identifier, key = "canonical", "parity"
    verification = entity(
        "check",
        "VERIFICATION",
        metadata={
            "method": "semantic_equivalence",
            "result": "passed",
            "observation_ids": ["a", "b"],
        },
    )
    if invalid == "empty_id":
        identifier = " "
    elif invalid == "single_observation":
        observations = (first,)
    elif invalid == "duplicate_ids":
        observations = (first, first)
    elif invalid == "different_semantic_key":
        observations = (first, replace(second, semantic_key="different"))
    elif invalid == "different_constraints":
        observations = (first, replace(second, constraints={"authorized": False}))
    elif invalid == "wrong_declared_key":
        key = "different"
    elif invalid == "unbound_verification":
        verification = replace(
            verification, metadata={**verification.metadata, "observation_ids": []}
        )
    else:
        verification = replace(verification, metadata={**verification.metadata, "result": "failed"})
    with pytest.raises(ValueError):
        CanonicalIdentity(identifier, key, observations, verification)


def test_direct_canonical_identity_freezes_observation_sequence_and_preserves_provenance():
    """Verify direct identity freezes observations and preserves provenance."""
    observations = [claim("a", semantic_key="parity"), claim("b", semantic_key="parity")]
    verification = entity(
        "check",
        "VERIFICATION",
        metadata={
            "method": "semantic_equivalence",
            "result": "passed",
            "observation_ids": ["a", "b"],
        },
    )
    canonical = CanonicalIdentity("canonical", "parity", observations, verification)
    observations.clear()
    assert len(canonical.observations) == 2
    assert len(canonical.provenance) == 2
