"""JSON-safe semantic constraint data types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ConstraintProvenance:
    """Source metadata for a bounded compiled constraint."""

    source_text: str
    source_span: tuple[int, int] | None = None
    compiler_name: str = "rule_based_shadow"
    compiler_version: str = "0.1"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SemanticConstraint:
    """A candidate-elimination constraint with provenance and verification state."""

    constraint_id: str
    relation: str
    subject: str
    object_value: object
    allowed_candidates: list[object]
    confidence: float
    provenance: ConstraintProvenance
    verified: bool = False
    verification_status: str = "unverified"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SemanticCompilationResult:
    """Structured result of a bounded semantic compilation pass."""

    input_text: str
    task_type: str
    constraints: list[SemanticConstraint]
    unsupported_fragments: list[str]
    ambiguity_detected: bool
    contradiction_detected: bool
    verification_status: str
    safe_for_shadow_projection: bool
    safe_for_gated_projection: bool
    diagnostics: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SemanticProjectionResult:
    """Projection payload for semantic constraints routed into finite lattices."""

    compilation: SemanticCompilationResult
    mode: str
    remaining_candidates: list[object]
    recommended_action: str
    lattice_constraints: list[dict[str, object]]
    gated: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = [
    "ConstraintProvenance",
    "SemanticCompilationResult",
    "SemanticConstraint",
    "SemanticProjectionResult",
]
