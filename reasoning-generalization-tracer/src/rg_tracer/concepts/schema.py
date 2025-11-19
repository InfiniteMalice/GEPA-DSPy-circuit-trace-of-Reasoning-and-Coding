"""Concept specification data structures."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping


@dataclass
class ConceptTest:
    """Simple input/output example used to validate a concept."""

    input: Mapping[str, Any]
    expected: Any

    def run(self, candidate: Callable[[Mapping[str, Any]], Any]) -> bool:
        try:
            return candidate(self.input) == self.expected
        except Exception:
            return False


@dataclass
class ConceptSpec:
    """Specification for a concept including tests and expected structures."""

    name: str
    definition: str
    tests: List[ConceptTest] = field(default_factory=list)
    expected_substructures: List[str] = field(default_factory=list)
    feature_catalog: List[Mapping[str, Any]] = field(default_factory=list)
    """Catalog of feature descriptors (id, tags, notes, etc.) exposed to attribution tools."""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConceptSpec":
        tests = [ConceptTest(**test) for test in data.get("tests", [])]
        raw_entries = data.get("feature_catalog")
        if raw_entries is None:
            source_entries: List[object] = []
        elif isinstance(raw_entries, Mapping):
            source_entries = [raw_entries]
        elif isinstance(raw_entries, Iterable) and not isinstance(raw_entries, (str, bytes)):
            source_entries = list(raw_entries)
        else:
            source_entries = []
        catalog = [dict(entry) for entry in source_entries if isinstance(entry, Mapping)]
        if len(catalog) < len(source_entries):
            dropped = len(source_entries) - len(catalog)
            warnings.warn(
                f"Dropped {dropped} invalid feature_catalog entries",
                RuntimeWarning,
                stacklevel=2,
            )
        validated_catalog: List[Mapping[str, Any]] = []
        for entry in catalog:
            identifier = entry.get("id")
            if identifier is None or str(identifier) == "":
                warnings.warn(
                    "feature_catalog entry missing usable 'id' field; skipping entry",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            validated_catalog.append(entry)
        return cls(
            name=data["name"],
            definition=data.get("definition", ""),
            tests=tests,
            expected_substructures=list(data.get("expected_substructures", [])),
            feature_catalog=validated_catalog,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "definition": self.definition,
            "tests": [{"input": test.input, "expected": test.expected} for test in self.tests],
            "expected_substructures": list(self.expected_substructures),
            "feature_catalog": [dict(entry) for entry in self.feature_catalog],
        }


__all__ = ["ConceptSpec", "ConceptTest"]
