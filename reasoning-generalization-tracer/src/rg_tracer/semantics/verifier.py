"""Lightweight semantic verifier detecting common reasoning pathologies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

from .taxonomy import SemanticTag


@dataclass
class SemanticReport:
    score: int
    contradiction_rate: float
    entailed_steps_pct: float
    fact_free_ratio: float
    unit_check_pass: bool
    symbol_binding_errors: int
    schema_consistency_pct: float
    tags: List[Dict[str, Iterable[str]]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "score": self.score,
            "contradiction_rate": self.contradiction_rate,
            "entailed_steps_pct": self.entailed_steps_pct,
            "fact_free_ratio": self.fact_free_ratio,
            "unit_check_pass": self.unit_check_pass,
            "symbol_binding_errors": self.symbol_binding_errors,
            "schema_consistency_pct": self.schema_consistency_pct,
            "tags": self.tags,
        }


_KEYWORDS_SUPPORT = {"because", "therefore", "thus", "since", "hence"}


def _normalise_chain(chain: object) -> List[str]:
    if isinstance(chain, str):
        steps = [step.strip() for step in chain.split("\n") if step.strip()]
    elif isinstance(chain, Mapping) and "steps" in chain:
        steps = [str(step).strip() for step in chain.get("steps", []) if str(step).strip()]
    elif isinstance(chain, Sequence):
        steps = [str(step).strip() for step in chain if str(step).strip()]
    else:
        steps = [str(chain).strip()]
    return steps or [""]


def _detect_contradiction(step: str) -> bool:
    tokens = step.lower()
    return "contradict" in tokens or "impossible" in tokens or "cannot" in tokens


def _detect_fact_free(step: str) -> bool:
    has_digit = any(char.isdigit() for char in step)
    has_equation = "=" in step or ":" in step
    has_support = any(keyword in step.lower() for keyword in _KEYWORDS_SUPPORT)
    return not (has_digit or has_equation or has_support)


def _detect_units(step: str, expected: str | None) -> bool:
    if not expected:
        return True
    lowered = step.lower()
    alt_units = {"meters", "seconds", "kg", "binary", "count", "mod"}
    mismatched = any(unit in lowered and unit != expected.lower() for unit in alt_units)
    if mismatched:
        return False
    if expected.lower() in lowered:
        return True
    return True


def _detect_variable_drift(step: str, allowed: set[str]) -> bool:
    tokens = {
        token.strip(".,:;!")
        for token in step.split()
        if token.isalpha() and len(token) == 1 and token.islower()
    }
    if not tokens:
        return False
    unexpected = {token for token in tokens if allowed and token not in allowed}
    return bool(unexpected)


def verify_chain(chain: object, problem_spec: Mapping[str, object]) -> SemanticReport:
    steps = _normalise_chain(chain)
    tags: List[Dict[str, List[str]]] = [{"tags": []} for _ in steps]
    expected_units = str(problem_spec.get("units", "")) or None
    allowed_vars = set(str(var) for var in problem_spec.get("variables", []))
    concept = str(problem_spec.get("concept", "")) or None

    contradiction_count = 0
    unsupported_count = 0
    entailed_count = 0
    schema_hits = 0
    variable_drift = 0
    unit_check_pass = True

    for idx, step in enumerate(steps):
        step_tags: List[str] = []
        if _detect_contradiction(step):
            step_tags.append(SemanticTag.CONTRADICTION.value)
            contradiction_count += 1
        fact_free = _detect_fact_free(step)
        if fact_free:
            step_tags.append(SemanticTag.UNSUPPORTED.value)
            unsupported_count += 1
        if concept and concept.lower() in step.lower():
            schema_hits += 1
        if expected_units:
            units_ok = _detect_units(step, expected_units)
            if not units_ok:
                step_tags.append(SemanticTag.UNIT_MISMATCH.value)
                unit_check_pass = False
        if allowed_vars and _detect_variable_drift(step, allowed_vars):
            step_tags.append(SemanticTag.VARIABLE_DRIFT.value)
            variable_drift += 1
        if not fact_free and SemanticTag.CONTRADICTION.value not in step_tags:
            step_tags.append(SemanticTag.ENTAILED.value)
            entailed_count += 1
        if not fact_free:
            step_tags.append(SemanticTag.SUPPORTED.value)
        tags[idx] = {"step": step, "tags": step_tags}

    total_steps = len(steps)
    contradiction_rate = contradiction_count / total_steps if total_steps else 0.0
    fact_free_ratio = unsupported_count / total_steps if total_steps else 0.0
    entailed_steps_pct = entailed_count / total_steps if total_steps else 0.0
    schema_consistency_pct = schema_hits / total_steps if total_steps else 1.0

    score = 4
    if contradiction_rate > 0 or not unit_check_pass:
        score = 0
    elif contradiction_rate > 0.1 or fact_free_ratio > 0.3:
        score = 1
    elif entailed_steps_pct < 0.5 or variable_drift > 1:
        score = 2
    elif entailed_steps_pct < 0.75 or fact_free_ratio > 0.2:
        score = 3
    if schema_consistency_pct < 0.5:
        score = min(score, 2)

    return SemanticReport(
        score=score,
        contradiction_rate=contradiction_rate,
        entailed_steps_pct=entailed_steps_pct,
        fact_free_ratio=fact_free_ratio,
        unit_check_pass=unit_check_pass,
        symbol_binding_errors=variable_drift,
        schema_consistency_pct=schema_consistency_pct,
        tags=tags,
    )


__all__ = ["SemanticReport", "verify_chain"]
