"""Conservative explicit equivalence; never infer semantic identity from surface similarity."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from .entities import Claim, OntologyEntity, Provenance
from .registry import EntityType


def candidate_equivalences(claims: Iterable[Claim]) -> tuple[tuple[str, str], ...]:
    """Suggest pairs only for explicit shared semantic keys and identical scoped constraints.

    The caller assigns semantic keys; matching keys are hypotheses, not proof of equivalence.
    """
    return tuple(
        (left.id, right.id)
        for left, right in combinations(claims, 2)
        if left.semantic_key
        and left.semantic_key == right.semantic_key
        and left.constraints == right.constraints
    )


@dataclass(frozen=True)
class CanonicalIdentity:
    """An identity grouping that preserves complete and potentially disagreeing observations."""

    id: str
    semantic_key: str
    observations: tuple[Claim, ...]
    verification: OntologyEntity

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", tuple(self.observations))
        _validate_identity(self.id, self.semantic_key, self.observations, self.verification)

    @property
    def provenance(self) -> tuple[Provenance, ...]:
        return tuple(item for claim in self.observations for item in claim.provenance)

    def to_dict(self):
        return {
            "id": self.id,
            "semantic_key": self.semantic_key,
            "observations": [claim.to_dict() for claim in self.observations],
            "verification": self.verification.to_dict(),
        }


def _validate_identity(
    canonical_id: str,
    semantic_key: str,
    observations: tuple[Claim, ...],
    verification: OntologyEntity,
) -> None:
    if not isinstance(canonical_id, str) or not canonical_id.strip() or len(observations) < 2:
        raise ValueError("Canonicalization requires an ID and at least two observations")
    if any(not isinstance(claim, Claim) for claim in observations):
        raise ValueError("Canonicalization observations must be Claim records")
    first = observations[0]
    if (
        not isinstance(semantic_key, str)
        or not semantic_key.strip()
        or any(
            claim.semantic_key != semantic_key or claim.constraints != first.constraints
            for claim in observations
        )
    ):
        raise ValueError(
            "Canonicalization requires matching explicit semantic keys and constraints"
        )
    if len({claim.id for claim in observations}) != len(observations):
        raise ValueError("Canonicalization requires distinct observation identities")
    if (
        not isinstance(verification, OntologyEntity)
        or verification.type != EntityType.VERIFICATION
        or verification.metadata.get("method") != "semantic_equivalence"
        or verification.metadata.get("result") != "passed"
        or set(verification.metadata.get("observation_ids", ())) != {c.id for c in observations}
    ):
        raise ValueError("Canonicalization requires an observation-bound equivalence verification")


def canonicalize_claims(
    claims: Iterable[Claim], canonical_id: str, verification: OntologyEntity
) -> CanonicalIdentity:
    """Group verified equivalent observations without merging evidence or promoting truth."""
    observations = tuple(claims)
    key = (
        observations[0].semantic_key if observations and isinstance(observations[0], Claim) else ""
    )
    return CanonicalIdentity(canonical_id, key, observations, verification)
