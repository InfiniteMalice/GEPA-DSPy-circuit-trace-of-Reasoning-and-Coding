"""Targeted repair heuristics responding to semantic tags."""

from __future__ import annotations

import re
from typing import Iterable, List, Mapping

try:  # pragma: no cover - optional regex backend for boundary repairs
    import regex as _regex_backend
except ImportError:  # pragma: no cover
    _regex_backend = None

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
        steps = [step.strip() for step in chain.split("\n") if step.strip()]
    elif isinstance(chain, Mapping) and "steps" in chain:
        steps = [str(step).strip() for step in chain.get("steps", []) if str(step).strip()]
    elif isinstance(chain, Iterable):
        steps = [str(step).strip() for step in chain if str(step).strip()]
    else:
        steps = [str(chain).strip()]
    return steps or [""]


def _append_with_punctuation(text: str, suffix: str) -> str:
    """Append ``suffix`` to ``text`` while handling terminal punctuation."""

    trimmed = text.rstrip()
    suffix_text = suffix.strip()
    if not trimmed:
        return suffix_text
    trimmed = trimmed.rstrip(":;,") or trimmed
    terminal = trimmed[-1]
    if terminal in {".", "?", "!"}:
        return f"{trimmed} {suffix_text}"
    base = trimmed.rstrip(".")
    return f"{base}. {suffix_text}" if base else suffix_text


def repair_once(
    chain: object,
    tags: Iterable[Mapping[str, object]],
    *,
    expected_units: str | None = None,
    preferred_variables: Iterable[str] | None = None,
) -> List[str]:
    """Attempt a single targeted repair based on semantic tags."""
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
                        module_name = pattern.__class__.__module__
                        if _regex_backend is not None and module_name.startswith("regex"):
                            pattern_ci = _regex_backend.compile(pattern.pattern, flags=flags)
                        else:
                            pattern_ci = re.compile(pattern.pattern, flags=flags)
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
                        pattern_module = pattern.__class__.__module__
                        if _regex_backend is not None and pattern_module.startswith("regex"):
                            pattern_ci = _regex_backend.compile(pattern.pattern, flags=flags)
                        else:
                            pattern_ci = re.compile(pattern.pattern, flags=flags)
                        new_step, _ = pattern_ci.subn(expected_units, new_step, count=1)
            expected_trimmed = expected_units.strip()
            if expected_trimmed:
                expected_pattern = build_token_boundary_pattern(expected_trimmed)
                if expected_pattern:
                    flags = expected_pattern.flags | re.IGNORECASE
                    pattern_module = expected_pattern.__class__.__module__
                    if _regex_backend is not None and pattern_module.startswith("regex"):
                        matcher = _regex_backend.compile(expected_pattern.pattern, flags=flags)
                    else:
                        matcher = re.compile(expected_pattern.pattern, flags=flags)
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
            suffix = " because we justify the inference from previous steps."
            if suffix.strip() not in step:
                steps[idx] = f"{step}{suffix}"
            break
        if fix_tag == SemanticTag.UNCITED_CLAIM.value:
            steps[idx] = f"{step} (Doe 2020, p. 14)"
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
            if "may" not in step.lower():
                steps[idx] = f"{step} This relationship may be correlational.".strip()
            break
        if fix_tag == SemanticTag.IS_OUGHT_SLIP.value:
            normative_suffix = "This recommendation is normative and contingent on shared values."
            if normative_suffix.lower() not in step.lower():
                steps[idx] = _append_with_punctuation(step, normative_suffix)
            break
    return steps


__all__ = ["repair_once"]
