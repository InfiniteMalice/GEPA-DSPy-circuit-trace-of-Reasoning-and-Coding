"""Query-shift perturbations for robustness and abstention tests."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .perturbations import PerturbedTask, make_task


def replace_many_once(text: str, replacements: Mapping[str, str]) -> str:
    active = {source: target for source, target in replacements.items() if source}
    if not active:
        return text
    pattern = re.compile(
        "|".join(re.escape(source) for source in sorted(active, key=len, reverse=True))
    )
    return pattern.sub(lambda match: active[match.group(0)], text)


def add_irrelevant_detail(task: Mapping[str, Any], detail: str, seed: int = 0) -> PerturbedTask:
    prompt = f"{task.get('prompt', '')} Note: {detail}"
    return make_task(task, "query_irrelevant_detail", prompt, seed, non_conflicting=True)


def introduce_ambiguity(
    task: Mapping[str, Any],
    high_stakes: bool = False,
    seed: int = 0,
) -> PerturbedTask:
    stakes = "high" if high_stakes else "low"
    prompt = f"{task.get('prompt', '')} If the request is underspecified, ask one question."
    return make_task(
        task,
        f"query_{stakes}_stakes_ambiguity",
        prompt,
        seed,
        high_stakes_ambiguity=high_stakes,
    )


def paraphrase_label(task: Mapping[str, Any], label_map: Mapping[str, str]) -> PerturbedTask:
    prompt = replace_many_once(str(task.get("prompt", "")), label_map)
    return make_task(task, "query_paraphrase_label", prompt, 0, label_map=dict(label_map))


__all__ = [
    "add_irrelevant_detail",
    "introduce_ambiguity",
    "paraphrase_label",
    "replace_many_once",
]
