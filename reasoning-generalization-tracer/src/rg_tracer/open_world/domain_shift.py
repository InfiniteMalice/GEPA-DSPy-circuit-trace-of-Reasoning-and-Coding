"""Domain and role-filler perturbations for pattern-matching checks."""

from __future__ import annotations

from typing import Any, Mapping

from .perturbations import PerturbedTask, make_task


def swap_surface_domain(
    task: Mapping[str, Any],
    source_domain: str,
    target_domain: str,
    seed: int = 0,
) -> PerturbedTask:
    prompt = str(task.get("prompt", "")).replace(source_domain, target_domain)
    return make_task(
        task,
        "domain_surface_swap",
        prompt,
        seed,
        source_domain=source_domain,
        target_domain=target_domain,
    )


def substitute_role_fillers(
    task: Mapping[str, Any],
    replacements: Mapping[str, str],
    seed: int = 0,
) -> PerturbedTask:
    prompt = str(task.get("prompt", ""))
    for source, target in replacements.items():
        prompt = prompt.replace(source, target)
    return make_task(task, "domain_role_filler", prompt, seed, replacements=dict(replacements))


__all__ = ["substitute_role_fillers", "swap_surface_domain"]
