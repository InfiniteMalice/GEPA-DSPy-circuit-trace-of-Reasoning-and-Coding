"""Dependency graph extraction from explicit skill manifests."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .manifest import SkillManifest


def dependency_graph(manifests: Iterable[SkillManifest]) -> Dict[str, List[str]]:
    graph: Dict[str, List[str]] = {}
    for manifest in manifests:
        graph[manifest.skill_id] = [
            dependency.name
            for dependency in manifest.dependencies
            if dependency.dependency_type == "skill"
        ]
    return graph


__all__ = ["dependency_graph"]
