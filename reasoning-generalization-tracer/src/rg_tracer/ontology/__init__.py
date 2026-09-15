"""Shared diagnostic identities for specialized reasoning and mechanistic graph views."""

from .canonicalization import CanonicalIdentity, candidate_equivalences, canonicalize_claims
from .entities import Claim, OntologyEntity, Provenance, Transformation
from .graph import OntologyGraph, OntologyRelation
from .registry import (
    ONTOLOGY_VERSION,
    EntityType,
    EpistemicStatus,
    GraphView,
    RelationType,
    load_registry,
)

__all__ = [
    "ONTOLOGY_VERSION",
    "CanonicalIdentity",
    "Claim",
    "EntityType",
    "EpistemicStatus",
    "GraphView",
    "OntologyEntity",
    "OntologyGraph",
    "OntologyRelation",
    "Provenance",
    "RelationType",
    "Transformation",
    "candidate_equivalences",
    "canonicalize_claims",
    "load_registry",
]
