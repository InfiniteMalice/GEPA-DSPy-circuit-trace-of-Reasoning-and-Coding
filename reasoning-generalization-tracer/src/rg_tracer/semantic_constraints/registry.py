"""Registry metadata for bounded semantic constraint compilers."""

from __future__ import annotations

from .compiler import RuleBasedSemanticConstraintCompiler


def available_compilers() -> dict[str, RuleBasedSemanticConstraintCompiler]:
    """Return deterministic compiler instances available in the default package."""

    compiler = RuleBasedSemanticConstraintCompiler()
    return {compiler.compiler_name: compiler}


__all__ = ["available_compilers"]
