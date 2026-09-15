"""Four typed graph views with distinct evidence and execution-lineage rules."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

from .entities import Claim, OntologyEntity, Provenance, freeze, serialize
from .registry import (
    ONTOLOGY_VERSION,
    EntityType,
    EpistemicStatus,
    GraphView,
    RelationType,
    _REGISTRY,
    relation_definition,
)


@dataclass(frozen=True)
class OntologyRelation:
    """An asserted directed edge; registry semantics determine its permitted endpoints."""

    source: str
    relation: RelationType | str
    target: str
    evidence_ids: tuple[str, ...] = ()
    provenance: tuple[Provenance, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize relation fields and require non-empty endpoints."""
        key = _REGISTRY["relation_aliases"].get(self.relation, self.relation)
        object.__setattr__(self, "relation", RelationType(key))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "metadata", freeze(self.metadata))
        if not self.source or not self.target:
            raise ValueError("Ontology relations require non-empty endpoints")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the value to a JSON-compatible dictionary."""
        return {
            "source": self.source,
            "relation": self.relation.value,
            "target": self.target,
            "evidence_ids": list(self.evidence_ids),
            "provenance": serialize(self.provenance),
            "metadata": serialize(self.metadata),
        }


class OntologyGraph:
    """Append identities and validated relations without modifying the source graphs."""

    def __init__(self, view: GraphView | str, *, metadata: Mapping[str, Any] | None = None):
        """Create an empty graph for ``view`` with immutable metadata."""
        self.view = GraphView(view)
        self.metadata = freeze(metadata or {})
        self._entities: dict[str, OntologyEntity] = {}
        self._relations: list[OntologyRelation] = []

    @property
    def entities(self) -> Mapping[str, OntologyEntity]:
        """Return the graph's entities."""
        return MappingProxyType(self._entities)

    @property
    def relations(self) -> tuple[OntologyRelation, ...]:
        """Return the graph's relations."""
        return tuple(self._relations)

    def add_entity(self, entity: OntologyEntity) -> None:
        """Add an entity while rejecting a conflicting duplicate identity."""
        previous = self._entities.get(entity.id)
        if previous is not None and previous != entity:
            raise ValueError(f"Entity identity already exists: {entity.id}")
        self._entities[entity.id] = entity

    def add_relation(self, relation: OntologyRelation) -> None:
        """Validate an edge and synchronize affected claims in one atomic insertion."""
        self._validate_relation(relation)
        proposed = [*self._relations, relation]
        if self.view == GraphView.EXECUTION:
            self._validate_dag(proposed)
        claim_updates = self._claim_updates(relation)
        self._relations.append(relation)
        self._entities.update(claim_updates)

    def _claim_updates(self, relation: OntologyRelation) -> dict[str, Claim]:
        """Keep task evidence fields synchronized without treating other claims as evidence."""
        source = self._entities[relation.source]
        target = self._entities[relation.target]
        if not isinstance(target, Claim) or relation.relation not in {
            RelationType.SUPPORTS,
            RelationType.CONTRADICTS,
        }:
            return {}
        contradicts = relation.relation == RelationType.CONTRADICTS
        changes: dict[str, Any] = {}
        if source.type == EntityType.EVIDENCE:
            name = "contradicting_evidence" if contradicts else "supporting_evidence"
            changes[name] = tuple(dict.fromkeys((*getattr(target, name), source.id)))
        targets = {target.id: (target, changes)}
        if contradicts and isinstance(source, Claim) and source.id != target.id:
            # A logical conflict disputes both claims without reversing the recorded assertion.
            targets[source.id] = (source, {})
        updates = {}
        for identifier, (claim, fields) in targets.items():
            if contradicts and claim.epistemic_status != EpistemicStatus.CONTRADICTED:
                fields["epistemic_status"] = EpistemicStatus.CONTRADICTED
                fields["status_history"] = (*claim.status_history, claim.epistemic_status.value)
            if fields:
                updates[identifier] = replace(claim, **fields)
        return updates

    def _is_disputed(self, claim: Claim) -> bool:
        """Inspect assertions as well as cached evidence to protect promotion and export."""
        return bool(claim.contradicting_evidence) or any(
            edge.relation == RelationType.CONTRADICTS
            and (
                edge.target == claim.id
                or (edge.source == claim.id and isinstance(self._entities[edge.target], Claim))
            )
            for edge in self._relations
        )

    def _validate_relation(self, relation: OntologyRelation) -> None:
        """Require relation endpoints and evidence to satisfy registry semantics."""
        try:
            source = self._entities[relation.source]
            target = self._entities[relation.target]
        except KeyError as error:
            raise ValueError(f"Missing relation endpoint: {error.args[0]}") from error
        definition = relation_definition(relation.relation)
        valid = any(
            signature["view"] == self.view.value
            and source.type.value in signature["domain"]
            and target.type.value in signature["range"]
            for signature in definition["signatures"]
        )
        if not valid or (definition.get("same_type") and source.type != target.type):
            raise ValueError(
                f"Invalid {self.view.value} relation: {source.type.value} "
                f"{relation.relation.value} {target.type.value}"
            )
        coordinate_fields = {
            "HAS_CASE": ("case_id", "case_id"),
            "HAS_STRIPE": ("stripe", "stripe_id"),
            "HAS_SUBTYPE": ("stripe_subtype", "subtype"),
            "HAS_REPEAT": ("repeat_id", "repeat_id"),
        }
        if relation.relation.value in coordinate_fields:
            source_field, target_field = coordinate_fields[relation.relation.value]
            if source.metadata.get(source_field) != target.metadata.get(target_field):
                raise ValueError("Evaluation relation contradicts its V5 coordinate")
            if relation.relation == RelationType.HAS_SUBTYPE:
                if source.metadata.get("stripe") != target.metadata.get("stripe_id"):
                    raise ValueError("Evaluation subtype belongs to a different stripe")
        for identifier in relation.evidence_ids:
            evidence = self._entities.get(identifier)
            if evidence is None or evidence.type.value not in {
                "EVIDENCE",
                "VERIFICATION",
                "MECHANISTIC_OBSERVATION",
            }:
                raise ValueError(f"Invalid relation evidence reference: {identifier}")
        if definition.get("requires_intervention"):
            tests = [self._entities[key] for key in relation.evidence_ids]
            if not any(
                test.type == EntityType.VERIFICATION
                and test.metadata.get("method") in {"intervention", "causal_test"}
                and test.metadata.get("result") == "passed"
                and test.metadata.get("source_id") == relation.source
                and test.metadata.get("target_id") == relation.target
                for test in tests
            ):
                raise ValueError("Causal relations require a passed, endpoint-bound causal test")

    def _validate_dag(self, relations: list[OntologyRelation]) -> None:
        """Reject cycles formed by relations whose registry definition requires a DAG."""
        adjacency: dict[str, set[str]] = {identifier: set() for identifier in self._entities}
        indegrees = dict.fromkeys(self._entities, 0)
        for relation in relations:
            source, target = relation.source, relation.target
            if relation_definition(relation.relation).get("execution_reverse"):
                source, target = target, source
            if target not in adjacency[source]:
                adjacency[source].add(target)
                indegrees[target] += 1
        pending = deque(key for key, count in indegrees.items() if count == 0)
        visited = 0
        while pending:
            source = pending.popleft()
            visited += 1
            for target in adjacency[source]:
                indegrees[target] -= 1
                if indegrees[target] == 0:
                    pending.append(target)
        if visited != len(self._entities):
            raise ValueError("Execution/provenance cycle detected")

    def record_evidence(self, claim_id: str, evidence_id: str, *, contradicts=False) -> None:
        """Append support or contradiction; preserve earlier evidence and status history."""
        claim = self._entities.get(claim_id)
        evidence = self._entities.get(evidence_id)
        if not isinstance(claim, Claim) or evidence is None or evidence.type != EntityType.EVIDENCE:
            raise ValueError("Claim evidence must reference a Claim and task-level Evidence")
        relation = "CONTRADICTS" if contradicts else "SUPPORTS"
        self.add_relation(OntologyRelation(evidence_id, relation, claim_id))

    def _task_verification(self, claim: Claim, verification_id: str) -> None:
        """Require a passed task verification bound to this claim."""
        verification = self._entities.get(verification_id)
        if (
            verification is None
            or verification.type != EntityType.VERIFICATION
            or verification.metadata.get("method") != "task_verification"
            or verification.metadata.get("result") != "passed"
            or verification.metadata.get("claim_id") != claim.id
        ):
            raise ValueError("Claim promotion requires a passed, claim-bound task verification")

    def promote_claim(self, claim_id: str, verification_id: str, *, established=False) -> None:
        """Record verification; this structural gate cannot judge a verifier's reliability."""
        claim = self._entities.get(claim_id)
        if not isinstance(claim, Claim) or not claim.supporting_evidence:
            raise ValueError("Claim promotion requires supporting task evidence")
        self._task_verification(claim, verification_id)
        if self._is_disputed(claim):
            raise ValueError("Resolve disputed claims in a new claim before promotion")
        self.add_relation(OntologyRelation(claim_id, "VERIFIED_BY", verification_id))
        status = EpistemicStatus.ESTABLISHED if established else EpistemicStatus.VERIFIED
        self._entities[claim_id] = replace(
            claim,
            epistemic_status=status,
            verification_ids=(*claim.verification_ids, verification_id),
            status_history=(*claim.status_history, claim.epistemic_status.value),
        )

    def validate(self) -> None:
        """Validate references and strong epistemic states before export or analysis."""
        for relation in self._relations:
            self._validate_relation(relation)
        if self.view == GraphView.EXECUTION:
            self._validate_dag(self._relations)
        for entity in self._entities.values():
            if not isinstance(entity, Claim):
                continue
            for key in (*entity.supporting_evidence, *entity.contradicting_evidence):
                evidence = self._entities.get(key)
                if evidence is None or evidence.type != EntityType.EVIDENCE:
                    raise ValueError(f"Invalid task evidence reference: {key}")
            if entity.epistemic_status.value in {"VERIFIED", "ESTABLISHED"}:
                if not entity.supporting_evidence or self._is_disputed(entity):
                    raise ValueError("Strong claim status requires undisputed supporting evidence")
                for key in entity.verification_ids:
                    self._task_verification(entity, key)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the value to a JSON-compatible dictionary."""
        self.validate()
        return {
            "ontology_version": ONTOLOGY_VERSION,
            "view": self.view.value,
            "entities": [entity.to_dict() for entity in self._entities.values()],
            "relations": [relation.to_dict() for relation in self._relations],
            "metadata": serialize(self.metadata),
        }
