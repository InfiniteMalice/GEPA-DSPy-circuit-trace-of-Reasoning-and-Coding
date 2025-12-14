"""Targeted repair heuristics responding to semantic tags."""

from __future__ import annotations

import re
from typing import Iterable, List, Mapping, Pattern

try:  # pragma: no cover - optional regex backend for boundary repairs
    import regex as _regex_backend
except ImportError:  # pragma: no cover
    _regex_backend = None

if _regex_backend is not None:  # pragma: no cover - optional dependency
    _REGEX_PATTERN_TYPE = getattr(_regex_backend, "Pattern", None)
else:  # pragma: no cover - regex not installed
    _REGEX_PATTERN_TYPE = None

from .patterns import build_token_boundary_pattern
from .taxonomy import SemanticTag


def _build_unit_pattern(unit_text: str) -> re.Pattern[str] | None:
    """Return a regex that respects token boundaries for ``unit_text``."""

    trimmed = unit_text.strip()
    if not trimmed:
        return None
    escaped = re.escape(trimmed)
    leading_alnum = trimmed[0].isalnum()
    trailing_alnum = trimmed[-1].isalnum()
    if leading_alnum and trailing_alnum:
        return re.compile(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])")
    return re.compile(escaped)


def _normalise_chain(chain: object) -> List[str]:
    if isinstance(chain, str):
        steps = [value for step in chain.split("\n") if (value := step.strip())]
    elif isinstance(chain, Mapping) and "steps" in chain:
        steps = [value for step in chain.get("steps", []) if (value := str(step).strip())]
    elif isinstance(chain, Iterable):
        steps = [value for step in chain if (value := str(step).strip())]
    else:
        steps = [str(chain).strip()]
    return steps or [""]


def _append_with_punctuation(text: str, suffix: str) -> str:
    """Append ``suffix`` to ``text`` while handling terminal punctuation."""

    trimmed = text.rstrip()
    suffix_text = suffix.strip()
    if not trimmed:
        return suffix_text
    # Preserve ellipses such as "..." without dropping existing periods.
    if set(trimmed) <= {"."}:
        return f"{trimmed} {suffix_text}"
    punctuation_only = all(char in ".:;," for char in trimmed)
    if punctuation_only:
        # Preserve ellipses ("..."), but drop bare separators such as ':' or ';'.
        return suffix_text if set(trimmed) - {"."} else f"{trimmed} {suffix_text}"
    trimmed = trimmed.rstrip(":;,") or trimmed
    terminal = trimmed[-1]
    if terminal in {".", "?", "!"}:
        return f"{trimmed} {suffix_text}"
    base = trimmed.rstrip(".")
    return f"{base}. {suffix_text}" if base else suffix_text


def _compile_case_insensitive(pattern: Pattern[str], *, flags: int) -> Pattern[str]:
    """Recompile ``pattern`` with ``flags`` while respecting optional regex backend."""

    if (
        _regex_backend is not None
        and _REGEX_PATTERN_TYPE is not None
        and isinstance(pattern, _REGEX_PATTERN_TYPE)
    ):
        return _regex_backend.compile(pattern.pattern, flags=flags)
    return re.compile(pattern.pattern, flags=flags)


def repair_once(
    chain: object,
    tags: Iterable[Mapping[str, object]],
    *,
    expected_units: str | None = None,
    preferred_variables: Iterable[str] | None = None,
) -> List[str]:
    """Attempt targeted repairs for a single pass of semantic tags.

    All tags other than ``UNIT_MISMATCH`` trigger at most one edit. Unit fixes may
    adjust multiple steps when the report identifies several concrete mismatches.
    """
    steps = _normalise_chain(chain)
    preferred = list(preferred_variables or [])
    replacement_var = preferred[0] if preferred else "x"
    tags_list = list(tags)
    priority = [
        SemanticTag.UNIT_MISMATCH.value,
        SemanticTag.VARIABLE_DRIFT.value,
        SemanticTag.UNSUPPORTED.value,
        SemanticTag.UNCITED_CLAIM.value,
        SemanticTag.MISQUOTE.value,
        SemanticTag.OVERCLAIMED_CAUSALITY.value,
        SemanticTag.IS_OUGHT_SLIP.value,
    ]
    fix_tag: str | None = None
    for candidate in priority:
        if candidate == SemanticTag.UNIT_MISMATCH.value and not expected_units:
            continue
        if any(candidate in entry.get("tags", []) for entry in tags_list):
            fix_tag = candidate
            break
    if fix_tag is None:
        return steps
    if fix_tag == SemanticTag.UNIT_MISMATCH.value:
        ordered_entries = sorted(
            tags_list,
            key=lambda item: bool(item.get("incorrect_unit")),
            reverse=True,
        )
    else:
        ordered_entries = tags_list
    for entry in ordered_entries:
        if fix_tag not in entry.get("tags", []):
            continue
        step = str(entry.get("step", "")).strip()
        try:
            idx = steps.index(step)
        except ValueError:
            continue
        if fix_tag == SemanticTag.VARIABLE_DRIFT.value:
            offending_token = str(entry.get("offending_token", "")).strip()
            candidate = offending_token
            if not candidate:
                match = re.search(r"\b([A-Za-z])\b", step)
                if match:
                    candidate = match.group(1)
            new_step = step
            if candidate:
                pattern = build_token_boundary_pattern(candidate)
                if pattern:
                    new_step, replaced = pattern.subn(replacement_var, new_step, count=1)
                    if replaced == 0:
                        flags = pattern.flags | re.IGNORECASE
                        pattern_ci = _compile_case_insensitive(pattern, flags=flags)
                        new_step, _ = pattern_ci.subn(replacement_var, new_step, count=1)
                else:
                    new_step = new_step.replace(candidate, replacement_var, 1)
            else:
                new_step = re.sub(r"\b[A-Za-z]\b", replacement_var, new_step, count=1)
            steps[idx] = new_step
            break
        if fix_tag == SemanticTag.UNIT_MISMATCH.value and expected_units:
            incorrect_unit = str(entry.get("incorrect_unit", "")).strip()
            new_step = step
            if incorrect_unit:
                pattern = build_token_boundary_pattern(incorrect_unit)
                if pattern:
                    new_step, replaced = pattern.subn(expected_units, new_step, count=1)
                    if replaced == 0:
                        flags = pattern.flags | re.IGNORECASE
                        pattern_ci = _compile_case_insensitive(pattern, flags=flags)
                        new_step, _ = pattern_ci.subn(expected_units, new_step, count=1)
            expected_trimmed = expected_units.strip()
            if expected_trimmed:
                expected_pattern = build_token_boundary_pattern(expected_trimmed)
                if expected_pattern:
                    flags = expected_pattern.flags | re.IGNORECASE
                    matcher = _compile_case_insensitive(expected_pattern, flags=flags)
                    has_expected = bool(matcher.search(new_step))
                else:
                    has_expected = expected_trimmed.lower() in new_step.lower()
                if not has_expected:
                    new_step = f"{new_step} ({expected_units})"
            steps[idx] = new_step
            if incorrect_unit:
                continue
            break
        if fix_tag == SemanticTag.UNSUPPORTED.value:
            suffix = "Because we justify the inference from previous steps."
            if suffix.lower() not in step.lower():
                steps[idx] = _append_with_punctuation(step, suffix)
            break
        if fix_tag == SemanticTag.UNCITED_CLAIM.value:
            citation = "(Doe 2020, p. 14)"
            if citation.lower() not in step.lower():
                steps[idx] = _append_with_punctuation(step, citation)
            break
        if fix_tag == SemanticTag.MISQUOTE.value:
            fixed = (
                step.replace("\u201c", '"')
                .replace("\u201d", '"')
                .replace("\u2018", "'")
                .replace("\u2019", "'")
                .strip()
            )
            if "[context clarified]" not in fixed.lower():
                fixed = f"{fixed} [context clarified]"
            steps[idx] = fixed
            break
        if fix_tag == SemanticTag.OVERCLAIMED_CAUSALITY.value:
            clarification = "This relationship may be correlational."
            if clarification.lower() not in step.lower():
                steps[idx] = _append_with_punctuation(step, clarification)
            break
        if fix_tag == SemanticTag.IS_OUGHT_SLIP.value:
            normative_suffix = "This recommendation is normative and contingent on shared values."
            if normative_suffix.lower() not in step.lower():
                steps[idx] = _append_with_punctuation(step, normative_suffix)
            break
    return steps


__all__ = ["repair_once"]
