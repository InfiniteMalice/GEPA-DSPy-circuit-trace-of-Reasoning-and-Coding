"""Lightweight semantic verifier detecting reasoning pathologies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

from .patterns import build_token_boundary_pattern, extract_letter_tokens
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
    humanities_metrics: Dict[str, float]
    tags: List[Dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        payload = {
            "score": self.score,
            "contradiction_rate": self.contradiction_rate,
            "entailed_steps_pct": self.entailed_steps_pct,
            "fact_free_ratio": self.fact_free_ratio,
            "unit_check_pass": self.unit_check_pass,
            "symbol_binding_errors": self.symbol_binding_errors,
            "schema_consistency_pct": self.schema_consistency_pct,
            "tags": self.tags,
        }
        payload.update(self.humanities_metrics)
        return payload


_KEYWORDS_SUPPORT = {"because", "therefore", "thus", "since", "hence"}
_ALT_UNITS = {"meters", "seconds", "kg", "binary", "count", "mod", "ternary"}


def _canonical_unit(value: str) -> str:
    trimmed = value.strip()
    if len(trimmed) > 2 and trimmed.endswith("s") and trimmed[-2].isalpha():
        return trimmed[:-1]
    return trimmed


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


def _detect_tag(step: str, substrings: Iterable[str]) -> bool:
    lowered = step.lower()
    return any(token in lowered for token in substrings)


def _detect_contradiction(step: str) -> bool:
    return _detect_tag(step, {"contradict", "impossible", "cannot"})


def _detect_fact_free(step: str) -> bool:
    has_digit = any(char.isdigit() for char in step)
    has_equation = "=" in step or ":" in step
    has_support = any(keyword in step.lower() for keyword in _KEYWORDS_SUPPORT)
    return not (has_digit or has_equation or has_support)


def _detect_units(step: str, expected: str | None) -> tuple[bool, str | None]:
    if not expected:
        return True, None
    lowered = step.lower()
    expected_lower = expected.lower().strip()
    if not expected_lower:
        return True, None
    pattern = build_token_boundary_pattern(expected_lower)
    matched_expected = bool(pattern and pattern.search(lowered))
    if matched_expected:
        return True, None
    canonical_expected = _canonical_unit(expected_lower)
    canonical_pattern = build_token_boundary_pattern(canonical_expected)
    matched_canonical = bool(canonical_pattern and canonical_pattern.search(lowered))
    if matched_canonical:
        return True, None
    tokens = extract_letter_tokens(lowered)
    for token in tokens:
        canonical_token = _canonical_unit(token)
        if canonical_token == canonical_expected:
            return True, None
    matched_variant = False
    detected_unit: str | None = None
    for unit in _ALT_UNITS:
        unit_lower = unit.lower().strip()
        if not unit_lower:
            continue
        canonical_detected = _canonical_unit(unit_lower)
        search_terms = {unit_lower}
        if canonical_detected:
            search_terms.add(canonical_detected)
        variant_match = False
        for term in search_terms:
            pattern = build_token_boundary_pattern(term)
            if pattern and pattern.search(lowered):
                variant_match = True
                break
        if not variant_match:
            continue
        candidate_unit = canonical_detected or unit_lower
        if canonical_detected == canonical_expected:
            matched_variant = True
            continue
        if detected_unit is None:
            detected_unit = candidate_unit
    if matched_variant:
        return True, None
    return False, detected_unit


def _detect_variable_drift(step: str, allowed: set[str]) -> tuple[bool, str | None]:
    tokens = set()
    for token in str(step).split():
        stripped = token.strip(".,:;!")
        normalised = stripped.casefold()
        if normalised.isalpha() and len(normalised) == 1:
            tokens.add(normalised)
    if not tokens:
        return False, None
    unexpected = sorted(token for token in tokens if allowed and token not in allowed)
    if not unexpected:
        return False, None
    return True, unexpected[0]


def _detect_humanities_tags(step: str, tags: List[str]) -> None:
    lowered = step.lower()
    if '"' in step and "(" not in step:
        tags.append(SemanticTag.MISQUOTE.value)
    if "according" in lowered and "(" not in step and "[" not in step:
        tags.append(SemanticTag.UNCITED_CLAIM.value)
    if "out of context" in lowered:
        tags.append(SemanticTag.QUOTE_OOC.value)
    if "therefore" in lowered and "because" not in lowered:
        tags.append(SemanticTag.ILLEGAL_INFERENCE.value)
    if "circular" in lowered or "same assumption" in lowered:
        tags.append(SemanticTag.CIRCULARITY.value)
    if "causes" in lowered and "control" not in lowered:
        tags.append(SemanticTag.OVERCLAIMED_CAUSALITY.value)
    if "should" in lowered and "evidence" in lowered:
        tags.append(SemanticTag.IS_OUGHT_SLIP.value)


def verify_chain(chain: object, problem_spec: Mapping[str, object]) -> SemanticReport:
    steps = _normalise_chain(chain)
    tags: List[Dict[str, object]] = [{"tags": []} for _ in steps]
    raw_units = problem_spec.get("units")
    expected_units = str(raw_units).strip() if raw_units is not None else None
    if expected_units == "":
        expected_units = None
    allowed_vars = {
        str(var).strip().casefold() for var in problem_spec.get("variables", []) if str(var).strip()
    }
    raw_concept = problem_spec.get("concept")
    concept = str(raw_concept).strip() if raw_concept is not None else None
    if concept == "":
        concept = None
    humanities_domain = problem_spec.get("domain") == "humanities"

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
        incorrect_unit: str | None = None
        if expected_units:
            units_ok, detected_unit = _detect_units(step, expected_units)
            if not units_ok:
                step_tags.append(SemanticTag.UNIT_MISMATCH.value)
                incorrect_unit = detected_unit
                unit_check_pass = False
        drift_detected, drift_token = _detect_variable_drift(step, allowed_vars)
        if drift_detected:
            step_tags.append(SemanticTag.VARIABLE_DRIFT.value)
            variable_drift += 1
        _detect_humanities_tags(step, step_tags)
        if humanities_domain and "definition" in step.lower():
            step_tags.append(SemanticTag.DEFINITION_DRIFT.value)
        if not fact_free and SemanticTag.CONTRADICTION.value not in step_tags:
            step_tags.append(SemanticTag.ENTAILED.value)
            entailed_count += 1
        if not fact_free:
            step_tags.append(SemanticTag.SUPPORTED.value)
        entry = {"step": step, "tags": step_tags}
        if incorrect_unit:
            entry["incorrect_unit"] = incorrect_unit
        if drift_token:
            entry["offending_token"] = drift_token
        tags[idx] = entry

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

    if humanities_domain:
        from ..humanities.signals import analyse_humanities_chain

        signals = analyse_humanities_chain(steps)
        humanities_metrics = {
            "citation_coverage": signals.citation_coverage,
            "quote_presence": signals.quote_presence,
            "counterevidence_ratio": signals.counterevidence_ratio,
            "hedge_rate": signals.hedge_rate,
            "fallacy_flags": signals.fallacy_flags,
            "neutrality_balance": signals.neutrality_balance,
        }
        if signals.tags:
            for idx, entry in enumerate(signals.tags):
                if idx < len(tags):
                    merged = list(tags[idx].get("tags", []))
                    merged.extend(entry.get("tags", []))
                    tags[idx]["tags"] = list(dict.fromkeys(merged))
    else:
        humanities_metrics = {
            "citation_coverage": 0.0,
            "quote_presence": 0.0,
            "counterevidence_ratio": 0.0,
            "hedge_rate": 0.0,
            "fallacy_flags": 0,
            "neutrality_balance": 0.0,
        }

    return SemanticReport(
        score=score,
        contradiction_rate=contradiction_rate,
        entailed_steps_pct=entailed_steps_pct,
        fact_free_ratio=fact_free_ratio,
        unit_check_pass=unit_check_pass,
        symbol_binding_errors=variable_drift,
        schema_consistency_pct=schema_consistency_pct,
        humanities_metrics=humanities_metrics,
        tags=tags,
    )


__all__ = ["SemanticReport", "verify_chain"]
