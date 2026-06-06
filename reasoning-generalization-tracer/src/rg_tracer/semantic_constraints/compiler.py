"""Rule-based bounded compiler for synthetic finite-domain requirements."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .types import ConstraintProvenance, SemanticCompilationResult, SemanticConstraint

_INT_RE = r"-?\d+"
_AMBIGUITY_PATTERN = re.compile(r"\b(maybe|perhaps|around|approximately|roughly|preferably)\b")


def _hashable_domain(domain: Sequence[object]) -> list[object]:
    candidates = []
    for item in domain:
        try:
            hash(item)
        except TypeError:
            continue
        candidates.append(item)
    return candidates


def _ints_from_text(text: str) -> list[int]:
    return [int(match) for match in re.findall(_INT_RE, text)]


class RuleBasedSemanticConstraintCompiler:
    """Bounded synthetic compiler for explicit finite-domain toy requirements."""

    compiler_name = "rule_based_shadow"
    compiler_version = "0.1"

    def compile(
        self,
        text: str,
        domain: Sequence[object],
        *,
        task_type: str = "semantic_constraint_toy",
    ) -> SemanticCompilationResult:
        """Compile only explicitly supported toy clauses."""

        domain_values = _hashable_domain(domain)
        normalized = text.strip()
        clauses = self._clauses(normalized)
        constraints: list[SemanticConstraint] = []
        unsupported = []
        ambiguity_detected = bool(_AMBIGUITY_PATTERN.search(normalized.lower()))
        for index, clause in enumerate(clauses):
            span = self._span_for_clause(normalized, clause)
            constraint = self._compile_clause(clause, domain_values, index, span)
            if constraint is None:
                unsupported.append(clause)
            else:
                constraints.append(constraint)
        return SemanticCompilationResult(
            input_text=text,
            task_type=task_type,
            constraints=constraints,
            unsupported_fragments=unsupported,
            ambiguity_detected=ambiguity_detected,
            contradiction_detected=False,
            verification_status="unverified",
            safe_for_shadow_projection=bool(constraints),
            safe_for_gated_projection=False,
            diagnostics={
                "compiler": self.compiler_name,
                "compiler_version": self.compiler_version,
                "domain_size": len(domain_values),
            },
        )

    def _clauses(self, text: str) -> list[str]:
        lowered = text.lower().strip().rstrip(".")
        lowered = lowered.replace("the answer must be ", "")
        lowered = lowered.replace("the answer must ", "")
        lowered = lowered.replace("answer must be ", "")
        lowered = lowered.replace("answer must ", "")
        parts = [part.strip(" .") for part in re.split(r"\band\b", lowered) if part.strip(" .")]
        return parts or ([lowered] if lowered else [])

    def _span_for_clause(self, text: str, clause: str) -> tuple[int, int] | None:
        start = text.lower().find(clause.lower())
        if start < 0:
            return None
        return start, start + len(clause)

    def _compile_clause(
        self,
        clause: str,
        domain: list[object],
        index: int,
        span: tuple[int, int] | None,
    ) -> SemanticConstraint | None:
        allowed: list[object]
        relation: str
        value: object
        if clause == "even":
            relation = "is_even"
            value = True
            allowed = [item for item in domain if isinstance(item, int) and item % 2 == 0]
        elif clause == "odd":
            relation = "is_odd"
            value = True
            allowed = [item for item in domain if isinstance(item, int) and item % 2 != 0]
        elif match := re.fullmatch(r"greater than (?P<value>-?\d+)", clause):
            relation = "greater_than"
            value = int(match.group("value"))
            allowed = [item for item in domain if isinstance(item, int) and item > value]
        elif match := re.fullmatch(r"greater than or equal to (?P<value>-?\d+)", clause):
            relation = "greater_than_or_equal"
            value = int(match.group("value"))
            allowed = [item for item in domain if isinstance(item, int) and item >= value]
        elif match := re.fullmatch(r"less than (?P<value>-?\d+)", clause):
            relation = "less_than"
            value = int(match.group("value"))
            allowed = [item for item in domain if isinstance(item, int) and item < value]
        elif match := re.fullmatch(r"less than or equal to (?P<value>-?\d+)", clause):
            relation = "less_than_or_equal"
            value = int(match.group("value"))
            allowed = [item for item in domain if isinstance(item, int) and item <= value]
        elif match := re.fullmatch(r"not be (?P<value>-?\d+)", clause):
            relation = "not_equal"
            value = int(match.group("value"))
            allowed = [item for item in domain if item != value]
        elif clause.startswith("one of "):
            relation = "one_of"
            values = _ints_from_text(clause)
            value = values
            allowed = [item for item in domain if item in values]
        else:
            return None
        provenance = ConstraintProvenance(
            source_text=clause,
            source_span=span,
            compiler_name=self.compiler_name,
            compiler_version=self.compiler_version,
        )
        return SemanticConstraint(
            constraint_id=f"semantic_{index}_{relation}",
            relation=relation,
            subject="answer",
            object_value=value,
            allowed_candidates=allowed,
            confidence=0.95,
            provenance=provenance,
        )


__all__ = ["RuleBasedSemanticConstraintCompiler"]
