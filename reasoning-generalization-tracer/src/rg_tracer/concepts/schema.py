"""Concept specification data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConceptSpec":
        tests = [ConceptTest(**test) for test in data.get("tests", [])]
        return cls(
            name=data["name"],
            definition=data.get("definition", ""),
            tests=tests,
            expected_substructures=list(data.get("expected_substructures", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "definition": self.definition,
            "tests": [{"input": test.input, "expected": test.expected} for test in self.tests],
            "expected_substructures": list(self.expected_substructures),
        }


__all__ = ["ConceptSpec", "ConceptTest"]
