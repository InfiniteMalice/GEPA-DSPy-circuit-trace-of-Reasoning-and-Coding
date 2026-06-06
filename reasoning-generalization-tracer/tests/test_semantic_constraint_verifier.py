"""Tests for semantic constraint verification."""

from __future__ import annotations

from rg_tracer.semantic_constraints import RuleBasedSemanticConstraintCompiler
from rg_tracer.semantic_constraints.verifier import verify_compilation_result


def _verify(text: str):
    compiler = RuleBasedSemanticConstraintCompiler()
    result = compiler.compile(text, [1, 2, 3, 4])
    return verify_compilation_result(result, [1, 2, 3, 4])


def test_duplicate_constraints_identified():
    result = _verify("The answer must be even and even.")
    assert "duplicate_constraint" in result.diagnostics["verification_errors"]
    assert not result.safe_for_gated_projection


def test_contradictions_identified():
    result = _verify("The answer must be even and odd.")
    assert result.contradiction_detected
    assert not result.safe_for_gated_projection


def test_unsupported_text_cannot_gate():
    result = _verify("The answer must be even and be blue.")
    assert result.unsupported_fragments == ["be blue"]
    assert not result.safe_for_gated_projection
