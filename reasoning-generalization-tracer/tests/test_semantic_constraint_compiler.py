"""Tests for bounded semantic constraint compilation."""

from __future__ import annotations

from rg_tracer.semantic_constraints import RuleBasedSemanticConstraintCompiler


def test_supported_clauses_compile_deterministically():
    compiler = RuleBasedSemanticConstraintCompiler()
    result = compiler.compile("The answer must be even and greater than 2.", [1, 2, 3, 4])
    assert [constraint.relation for constraint in result.constraints] == [
        "is_even",
        "greater_than",
    ]
    assert result.constraints[0].allowed_candidates == [2, 4]


def test_unsupported_fragments_are_surfaced():
    compiler = RuleBasedSemanticConstraintCompiler()
    result = compiler.compile("The answer must be prime.", [1, 2, 3, 4])
    assert result.constraints == []
    assert result.unsupported_fragments == ["prime"]


def test_provenance_and_confidence_are_retained():
    compiler = RuleBasedSemanticConstraintCompiler()
    result = compiler.compile("The answer must not be 3.", [1, 2, 3, 4])
    constraint = result.constraints[0]
    assert constraint.provenance.source_text == "not be 3"
    assert 0.0 <= constraint.confidence <= 1.0


def test_ambiguous_wording_detected():
    compiler = RuleBasedSemanticConstraintCompiler()
    result = compiler.compile("The answer must maybe be even.", [1, 2, 3, 4])
    assert result.ambiguity_detected


def test_greater_than_or_equal_is_not_marked_ambiguous():
    compiler = RuleBasedSemanticConstraintCompiler()
    result = compiler.compile("The answer must be greater than or equal to 2.", [1, 2, 3, 4])
    assert not result.ambiguity_detected
    assert [constraint.relation for constraint in result.constraints] == ["greater_than_or_equal"]
    assert result.constraints[0].allowed_candidates == [2, 3, 4]
