"""Deterministic scoring axes for reasoning fitness.

Schema::
{
  "reasoning_axes": {
    "logical_validity": {
      "score": 0,
      "contradictions": 0,
      "formal_proof": false
    },
    "conceptual_clarity": {
      "score": 0,
      "undefined_symbols": 0
    },
    "completeness": {
      "score": 0,
      "edge_cases": 0
    },
    "rigor": {
      "score": 0,
      "checked_steps": 0
    },
    "efficiency": {
      "score": 0,
      "steps": 0
    },
    "heuristic_creativity": {
      "score": 0,
      "distinct_paths": 0
    },
    "numerical_accuracy": {
      "score": 0,
      "error_tolerance": 0
    },
    "cognitive_efficiency": {
      "score": 0,
      "token_count": 0,
      "time_ms": 0
    },
    "explanatory_power": {
      "score": 0,
      "causal_links": 0
    },
    "self_consistency": {
      "score": 0,
      "self_corrections": 0
    },
    "abstraction_generalization": {
      "score": 0,
      "transfer_accuracy": 0.0,
      "compression_gain": 0.0,
      "variable_lifts": 0,
      "theorem_induced": 0
    }
  }
}
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping


def _as_int(score: Any) -> int | None:
    """Return the integer override when ``score`` already sits in the 0–4 range."""
    if isinstance(score, bool):
        return None
    if isinstance(score, int) and 0 <= score <= 4:
        return score
    return None


def _bounded(value: float, minimum: int = 0, maximum: int = 4) -> int:
    return max(minimum, min(int(round(value)), maximum))


def logical_validity(metrics: Mapping[str, Any]) -> int:
    """Anchor: 0 = contradiction; 4 = contradiction-free formal proof."""
    override = _as_int(metrics.get("score"))
    if override is not None:
        return override
    contradictions = int(metrics.get("contradictions", 0) or 0)
    if contradictions >= 2:
        return 0
    if contradictions == 1:
        return 1
    if metrics.get("formal_proof"):
        return 4
    proof_like = metrics.get("proof_like")
    if isinstance(proof_like, (int, float)) and proof_like >= 0.8:
        return 3
    return 2


def conceptual_clarity(metrics: Mapping[str, Any]) -> int:
    """Anchor: 0 = undefined symbols everywhere; 4 = fully typed and consistent."""
    override = _as_int(metrics.get("score"))
    if override is not None:
        return override
    undefined = int(metrics.get("undefined_symbols", 0) or 0)
    if undefined > 3:
        return 0
    if undefined > 1:
        return 1
    if undefined == 1:
        return 2
    jargon = int(metrics.get("jargon_flags", 0) or 0)
    return 4 if jargon == 0 else 3


def completeness(metrics: Mapping[str, Any]) -> int:
    """Anchor: 0 = ignores boundaries; 4 = exhaustive coverage with edge checks."""
    override = _as_int(metrics.get("score"))
    if override is not None:
        return override
    missing = int(metrics.get("edge_cases", 0) or 0)
    if missing >= 3:
        return 0
    if missing == 2:
        return 1
    if missing == 1:
        return 2
    coverage = metrics.get("coverage_ratio", 1.0)
    if isinstance(coverage, (int, float)) and coverage < 0.9:
        return 3
    return 4


def rigor(metrics: Mapping[str, Any]) -> int:
    """Anchor: 0 = unchecked; 4 = ≥95% steps justified with symbolic checks."""
    override = _as_int(metrics.get("score"))
    if override is not None:
        return override
    checked = int(metrics.get("checked_steps", 0) or 0)
    total = int(metrics.get("total_steps", checked) or checked)
    if total == 0:
        return 0
    ratio = checked / total
    if ratio >= 0.95 and total >= 3:
        return 4
    if ratio >= 0.75:
        return 3
    if ratio >= 0.5:
        return 2
    return 1


def efficiency(metrics: Mapping[str, Any], baseline_steps: int | None = None) -> int:
    """Anchor: 0 = bloated; 4 = ≤50% of baseline steps."""
    override = _as_int(metrics.get("score"))
    if override is not None:
        return override
    steps = int(metrics.get("steps", 0) or 0)
    if baseline_steps is None:
        baseline_steps = int(metrics.get("baseline_steps", steps or 1) or 1)
    if baseline_steps <= 0:
        baseline_steps = 1
    if steps == 0:
        return 0
    ratio = steps / baseline_steps
    if ratio <= 0.5:
        return 4
    if ratio <= 0.8:
        return 3
    if ratio <= 1.2:
        return 2
    if ratio <= 1.5:
        return 1
    return 0


def heuristic_creativity(metrics: Mapping[str, Any]) -> int:
    """Anchor: 0 = rote path; 4 = ≥3 distinct, novel strategies."""
    override = _as_int(metrics.get("score"))
    if override is not None:
        return override
    paths = int(metrics.get("distinct_paths", 0) or 0)
    novelty = metrics.get("novelty", 0.0) or 0.0
    if paths == 0 and novelty < 0.1:
        return 0
    if paths <= 1 and novelty < 0.2:
        return 1
    if paths <= 2 and novelty < 0.4:
        return 2
    if paths >= 3 and novelty >= 0.5:
        return 4
    return 3


def numerical_accuracy(metrics: Mapping[str, Any]) -> int:
    """Anchor: 0 = ≥50% error; 4 = within half tolerance of true value."""
    override = _as_int(metrics.get("score"))
    if override is not None:
        return override
    tolerance = metrics.get("error_tolerance", 0.0) or 0.0
    if not isfinite(float(tolerance)):
        return 0
    error_rate = metrics.get("error_rate", 0.0) or 0.0
    if error_rate >= 0.5:
        return 0
    if error_rate >= 0.2:
        return 1
    if error_rate >= tolerance:
        return 2
    if error_rate <= tolerance / 2:
        return 4
    return 3


def cognitive_efficiency(metrics: Mapping[str, Any]) -> int:
    """Anchor: 0 = blows budgets; 4 = well within token/time/memory limits."""
    override = _as_int(metrics.get("score"))
    if override is not None:
        return override
    tokens = int(metrics.get("token_count", 0) or 0)
    time_ms = float(metrics.get("time_ms", 0.0) or 0.0)
    memory = float(metrics.get("memory_mb", 0.0) or 0.0)
    if tokens == 0 and time_ms == 0:
        return 0
    penalty = 0
    if tokens > metrics.get("token_budget", 512):
        penalty += 1
    if time_ms > metrics.get("time_budget_ms", 2000):
        penalty += 1
    if memory > metrics.get("memory_budget_mb", 256):
        penalty += 1
    return max(0, 4 - penalty)


def explanatory_power(metrics: Mapping[str, Any]) -> int:
    """Anchor: 0 = opaque; 4 = multi-causal explanation with examples."""
    override = _as_int(metrics.get("score"))
    if override is not None:
        return override
    causal = int(metrics.get("causal_links", 0) or 0)
    examples = int(metrics.get("examples", 0) or 0)
    if causal == 0 and examples == 0:
        return 0
    if causal < 2:
        return 2 if examples else 1
    if causal >= 4 and examples >= 2:
        return 4
    return 3


def self_consistency(metrics: Mapping[str, Any]) -> int:
    """Anchor: 0 = self-contradictory; 4 = high agreement with self-repair."""
    override = _as_int(metrics.get("score"))
    if override is not None:
        return override
    corrections = int(metrics.get("self_corrections", 0) or 0)
    if corrections > 3:
        return 0
    if corrections > 1:
        return 1
    if corrections == 1:
        return 2
    agreement = metrics.get("agreement_rate", 1.0) or 1.0
    if agreement >= 0.95:
        return 4
    if agreement >= 0.85:
        return 3
    return 2


def abstraction_generalization(metrics: Mapping[str, Any]) -> int:
    """Anchor: 0 = no transfer; 4 = high transfer, compression, and induced lemmas."""
    override = _as_int(metrics.get("score"))
    if override is not None:
        return override
    transfer = metrics.get("transfer_accuracy", 0.0) or 0.0
    compression = metrics.get("compression_gain", 0.0) or 0.0
    lifts = int(metrics.get("variable_lifts", 0) or 0)
    theorems = int(metrics.get("theorem_induced", 0) or 0)
    score = 0
    if transfer >= 0.9 and compression >= 0.2 and lifts >= 2:
        score = 4
    elif transfer >= 0.75 and compression >= 0.1:
        score = 3
    elif transfer >= 0.6 or lifts >= 1:
        score = 2
    elif transfer >= 0.4 or compression > 0.0 or theorems > 0:
        score = 1
    else:
        score = 0
    if score >= 2 and theorems >= 2:
        score = min(4, score + 1)
    return score


__all__ = [
    "logical_validity",
    "conceptual_clarity",
    "completeness",
    "rigor",
    "efficiency",
    "heuristic_creativity",
    "numerical_accuracy",
    "cognitive_efficiency",
    "explanatory_power",
    "self_consistency",
    "abstraction_generalization",
]
