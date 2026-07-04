"""Simple supply-chain risk warnings for explicit skill manifests."""

from __future__ import annotations

from typing import List

from .dependencies import dependency_graph
from .manifest import SkillManifest


def skill_risk_warnings(manifest: SkillManifest) -> List[str]:
    warnings: List[str] = []
    if not manifest.version:
        warnings.append("missing version")
    if not manifest.provenance:
        warnings.append("missing provenance")
    for dependency in manifest.dependencies:
        if not dependency.version:
            warnings.append(f"dependency {dependency.name} missing version")
        if not dependency.source:
            warnings.append(f"dependency {dependency.name} missing source")
        if dependency.dependency_type == "service":
            warnings.append(f"external service dependency: {dependency.name}")
    return warnings


def recursive_skill_warnings(manifests: list[SkillManifest]) -> List[str]:
    graph = dependency_graph(manifests)
    warnings: List[str] = []
    for skill_id in graph:
        if _has_cycle(skill_id, skill_id, graph, set()):
            warnings.append(f"recursive skill dependency: {skill_id}")
    return warnings


def _has_cycle(start: str, node: str, graph: dict[str, list[str]], seen: set[str]) -> bool:
    for target in graph.get(node, []):
        if target == start:
            return True
        if target not in seen:
            seen.add(target)
            if _has_cycle(start, target, graph, seen):
                return True
    return False


__all__ = ["recursive_skill_warnings", "skill_risk_warnings"]
