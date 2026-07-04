"""Explicit skill manifests for optional reusable skills."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass
class SkillDependency:
    dependency_type: str
    name: str
    version: str | None = None
    source: str | None = None
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dependency_type": self.dependency_type,
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "required": self.required,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SkillDependency":
        return cls(
            dependency_type=str(data["dependency_type"]),
            name=str(data["name"]),
            version=data.get("version"),
            source=data.get("source"),
            required=bool(data.get("required", True)),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class SkillManifest:
    skill_id: str
    name: str
    description: str
    version: str | None
    entrypoint: str
    dependencies: List[SkillDependency] = field(default_factory=list)
    provenance: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "entrypoint": self.entrypoint,
            "dependencies": [dependency.to_dict() for dependency in self.dependencies],
            "provenance": self.provenance,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SkillManifest":
        return cls(
            skill_id=str(data["skill_id"]),
            name=str(data["name"]),
            description=str(data.get("description", "")),
            version=data.get("version"),
            entrypoint=str(data.get("entrypoint", "")),
            dependencies=[
                SkillDependency.from_mapping(item) for item in data.get("dependencies", [])
            ],
            provenance=data.get("provenance"),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "SkillManifest":
        return cls.from_mapping(json.loads(text))


__all__ = ["SkillDependency", "SkillManifest"]
