"""Targeted repair heuristics responding to semantic tags."""

from __future__ import annotations

from typing import Iterable, List, Mapping

from .taxonomy import SemanticTag


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


def repair_once(
    chain: object,
    tags: Iterable[Mapping[str, Iterable[str]]],
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
    for entry in tags_list:
        if fix_tag not in entry.get("tags", []):
            continue
        step = str(entry.get("step", ""))
        try:
            idx = steps.index(step)
        except ValueError:
            continue
        if fix_tag == SemanticTag.VARIABLE_DRIFT.value:
            steps[idx] = step.replace("y", replacement_var)
            break
        if fix_tag == SemanticTag.UNIT_MISMATCH.value and expected_units:
            replacements = {"meters", "meter", "seconds", "second", "kg", "kilogram"}
            new_step = step
            for token in replacements:
                if token in new_step:
                    new_step = new_step.replace(token, expected_units)
            if expected_units not in new_step:
                new_step = f"{new_step} ({expected_units})"
            steps[idx] = new_step
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
            steps[idx] = step.replace('"', '"').strip()
            if "context" not in steps[idx].lower():
                steps[idx] = f"{steps[idx]} [context clarified]"
            break
        if fix_tag == SemanticTag.OVERCLAIMED_CAUSALITY.value:
            if "may" not in step.lower():
                steps[idx] = f"{step} This relationship may be correlational.".strip()
            break
        if fix_tag == SemanticTag.IS_OUGHT_SLIP.value:
            steps[idx] = f"{step} This recommendation is normative and contingent on shared values."
            break
    return steps


__all__ = ["repair_once"]
