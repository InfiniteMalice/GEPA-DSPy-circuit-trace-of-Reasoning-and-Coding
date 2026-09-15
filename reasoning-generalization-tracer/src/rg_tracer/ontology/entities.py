"""Immutable shared identities; richer specialized graph objects remain independent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

from .registry import EntityType, EpistemicStatus, _REGISTRY


def freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze(item) for item in value)
    return value


def serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset)):
        return [serialize(item) for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


@dataclass(frozen=True)
class Provenance:
    """A source observation; absent optional values stay unknown rather than fabricated."""

    source: str
    source_span: str | None = None
    run_id: str | None = None
    step_id: str | None = None
    created_by: str | None = None
    observation_time: str | None = None
    transformation: str | None = None
    verification: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Provenance requires a non-empty source")

    def to_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True)
class OntologyEntity:
    """Stable identity with source records; metadata is a frozen defensive copy."""

    id: str
    type: EntityType | str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[Provenance, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Entity IDs must be non-empty strings")
        if not isinstance(self.canonical_name, str) or not self.canonical_name.strip():
            raise ValueError("Entities require a stable non-empty canonical_name")
        object.__setattr__(self, "type", EntityType(self.type))
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "metadata", freeze(self.metadata))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        if self.type.value in _REGISTRY["provenance_required"] and not self.provenance:
            raise ValueError(f"{self.type.value} requires provenance")
        if self.type == EntityType.CLAIM and not isinstance(self, Claim):
            raise ValueError("Use Claim for CLAIM entities to preserve epistemic fields")
        if any(not isinstance(item, Provenance) for item in self.provenance):
            raise ValueError("provenance must contain Provenance records")
        if self.type == EntityType.CANONICAL_CASE:
            from ..epistemic_cases import FRAMEWORK_VERSION, get_case

            case = get_case(self.metadata.get("case_id"))
            if (
                self.id != f"{FRAMEWORK_VERSION}:case:{case.id}"
                or self.canonical_name != case.title
            ):
                raise ValueError("Canonical case identity must resolve through the V5 contract")
        if self.type in {EntityType.ROBUSTNESS_STRIPE, EntityType.STRIPE_SUBTYPE}:
            self._validate_stripe_identity()

    def _validate_stripe_identity(self) -> None:
        from ..epistemic_cases import FRAMEWORK_VERSION, stripe_registry, validate_coordinate

        stripe = self.metadata.get("stripe_id")
        subtype = self.metadata.get("subtype")
        validate_coordinate(0, stripe, subtype)
        if self.type == EntityType.ROBUSTNESS_STRIPE:
            identifier = f"{FRAMEWORK_VERSION}:stripe:{stripe}"
            name = stripe_registry()[stripe]["title"]
        else:
            if subtype is None:
                raise ValueError("StripeSubtype requires a registered subtype")
            identifier = f"{FRAMEWORK_VERSION}:subtype:{stripe}:{subtype}"
            name = subtype
        if self.id != identifier or self.canonical_name != name:
            raise ValueError("Stripe identity must resolve through the V5 contract")

    def to_dict(self) -> dict[str, Any]:
        return {item.name: serialize(getattr(self, item.name)) for item in fields(self)}


@dataclass(frozen=True)
class Claim(OntologyEntity):
    """A generated claim defaults to INFERRED; rich content does not require a triple."""

    type: EntityType = field(default=EntityType.CLAIM, init=False)
    subject: Any = None
    predicate: str | None = None
    object: Any = None
    content: Any = None
    epistemic_status: EpistemicStatus | str = EpistemicStatus.INFERRED
    confidence: float | None = None
    confidence_source: str | None = None
    semantic_key: str | None = None
    constraints: Mapping[str, Any] = field(default_factory=dict)
    supporting_evidence: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    verification_ids: tuple[str, ...] = ()
    status_history: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "epistemic_status", EpistemicStatus(self.epistemic_status))
        for name in ("subject", "object", "content", "constraints"):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        for name in (
            "supporting_evidence",
            "contradicting_evidence",
            "verification_ids",
            "status_history",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.confidence is not None:
            if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
                raise ValueError("Claim confidence must be finite and between zero and one")
            if not self.confidence_source:
                raise ValueError("Claim confidence requires confidence_source")
        if self.epistemic_status.value in {"VERIFIED", "ESTABLISHED"}:
            if not self.verification_ids:
                raise ValueError("Strong claim status requires verification references")


@dataclass(frozen=True)
class Transformation(OntologyEntity):
    """A reported transformation, including failed invariants and group-overlay lineage."""

    type: EntityType = field(default=EntityType.TRANSFORMATION, init=False)
    kind: str = "RepresentationChange"
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    preserved_properties: tuple[str, ...] = ()
    changed_properties: tuple[str, ...] = ()
    expected_invariants: tuple[str, ...] = ()
    observed_invariants: tuple[str, ...] = ()
    symmetry_breaks: tuple[str, ...] = ()
    inverse: str | None = None
    composition_parent: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind not in _REGISTRY["transformation_kinds"]:
            raise ValueError(f"Unregistered transformation kind: {self.kind}")
        for name in (
            "inputs",
            "outputs",
            "preserved_properties",
            "changed_properties",
            "expected_invariants",
            "observed_invariants",
            "symmetry_breaks",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
