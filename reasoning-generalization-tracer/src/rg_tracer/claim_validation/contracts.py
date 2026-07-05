"""Validation contracts for claim-boundary checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ValidationContract:
    contract_id: str
    task_id: str
    claim: str
    required_evidence: List[str] = field(default_factory=list)
    required_tests: List[str] = field(default_factory=list)
    forbidden_shortcuts: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "task_id": self.task_id,
            "claim": self.claim,
            "required_evidence": list(self.required_evidence),
            "required_tests": list(self.required_tests),
            "forbidden_shortcuts": list(self.forbidden_shortcuts),
            "acceptance_criteria": list(self.acceptance_criteria),
            "metadata": dict(self.metadata),
        }


def validate_contract(contract: ValidationContract, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    evidence = diagnostics.get("evidence") or []
    tests = diagnostics.get("tests") or []
    shortcuts = diagnostics.get("shortcuts") or []
    missing_evidence = [item for item in contract.required_evidence if item not in evidence]
    missing_tests = [item for item in contract.required_tests if item not in tests]
    shortcut_hits = [item for item in contract.forbidden_shortcuts if item in shortcuts]
    passed = not missing_evidence and not missing_tests and not shortcut_hits
    return {
        "passed": passed,
        "missing_evidence": missing_evidence,
        "missing_tests": missing_tests,
        "forbidden_shortcuts": shortcut_hits,
    }


__all__ = ["ValidationContract", "validate_contract"]
