"""DSPy signatures or import-safe stubs for schema V3 pipeline integration."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

if importlib.util.find_spec("dspy") is not None:
    dspy = importlib.import_module("dspy")
    Sig = dspy.Signature
else:

    class _Field:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class _DspyStub:
        class Signature:
            pass

        @staticmethod
        def InputField(**kwargs: Any) -> _Field:
            return _Field(**kwargs)

        @staticmethod
        def OutputField(**kwargs: Any) -> _Field:
            return _Field(**kwargs)

    dspy = _DspyStub()
    Sig = dspy.Signature


class ClassifyReasoningUnits(Sig):
    """Classify public reasoning units required and observed for a task."""

    task: str = dspy.InputField()
    answer: str = dspy.InputField()
    required_units: list[str] = dspy.OutputField()
    observed_units: list[str] = dspy.OutputField()


class SelectReasoningUnits(Sig):
    """Select compositional reasoning units before solving."""

    task: str = dspy.InputField()
    constraints: list[str] = dspy.InputField()
    selected_units: list[str] = dspy.OutputField()


class FrameTask(Sig):
    """Frame the task, stakes, and answer mode."""

    task: str = dspy.InputField()
    frame: dict[str, Any] = dspy.OutputField()


class DetermineGrounding(Sig):
    """Determine whether evidence and provenance ground the answer."""

    claim: str = dspy.InputField()
    evidence: list[str] = dspy.InputField()
    grounding_status: str = dspy.OutputField()


class SelectMethod(Sig):
    """Select the method appropriate to the task and stakes."""

    task: str = dspy.InputField()
    candidate_methods: list[str] = dspy.InputField()
    selected_method: str = dspy.OutputField()


class TrackAssumptions(Sig):
    """Track assumptions used by a public solution path."""

    task: str = dspy.InputField()
    assumptions: list[str] = dspy.OutputField()


class EstimateUncertainty(Sig):
    """Estimate uncertainty and evidence gaps."""

    claim: str = dspy.InputField()
    evidence: list[str] = dspy.InputField()
    confidence: float = dspy.OutputField()
    uncertainty_drivers: list[str] = dspy.OutputField()


class CalibrationDecision(Sig):
    """Decide whether confidence should be held, pushed up, or pushed down."""

    confidence: float = dspy.InputField()
    evidence_status: str = dspy.InputField()
    calibration_status: str = dspy.OutputField()


class ScientificMethodCheck(Sig):
    """Check causal/scientific claims using public scientific controls."""

    claim: str = dspy.InputField()
    evidence: list[str] = dspy.InputField()
    confounders: list[str] = dspy.OutputField()
    falsification_conditions: list[str] = dspy.OutputField()


class MDLControlGate(Sig):
    """Compare fast/default and controlled answers and decide escalation."""

    default_answer: str = dspy.InputField()
    controlled_answer: str = dspy.InputField()
    escalation_required: bool = dspy.OutputField()
    compression_guardrails: list[str] = dspy.OutputField()


class UnitTraceEvaluator(Sig):
    """Evaluate public unit traces without using private hidden chain-of-thought."""

    required_units: list[str] = dspy.InputField()
    observed_units: list[str] = dspy.InputField()
    missing_units: list[str] = dspy.OutputField()
    failed_units: list[str] = dspy.OutputField()


class CaseV3Classifier(Sig):
    """Classify a response into base case plus V3 overlays."""

    output_text: str = dspy.InputField()
    expected_answer: str | None = dspy.InputField()
    case_id: int = dspy.OutputField()
    compact_label: str = dspy.OutputField()


class DetectInvariantsUnderTransformation(Sig):
    """Identify what properties remain invariant under a transformation."""

    original: str = dspy.InputField()
    transformed: str = dspy.InputField()
    target_property: str = dspy.InputField()
    invariant_properties: list[str] = dspy.OutputField()
    changed_properties: list[str] = dspy.OutputField()


class ClassifyEquivalenceClass(Sig):
    """Group variants into equivalence classes based on preserved structure."""

    variants: list[str] = dspy.InputField()
    equivalence_criterion: str = dspy.InputField()
    classes: list[dict[str, Any]] = dspy.OutputField()
    symmetry_breaks: list[str] = dspy.OutputField()


class CanonicalizeRepresentation(Sig):
    """Convert surface-different variants into a canonical representation."""

    input_text: str = dspy.InputField()
    canonical_schema: str = dspy.InputField()
    canonical_form: dict[str, Any] = dspy.OutputField()
    lost_information: list[str] = dspy.OutputField()


class DetectSymmetryBreak(Sig):
    """Detect when two apparently similar cases differ in a relevant way."""

    case_a: str = dspy.InputField()
    case_b: str = dspy.InputField()
    proposed_symmetry: str = dspy.InputField()
    symmetry_holds: bool = dspy.OutputField()
    symmetry_breaking_factors: list[str] = dspy.OutputField()


class GenerateTransformationOrbit(Sig):
    """Generate or analyze variants reachable under allowed transformations."""

    seed_case: str = dspy.InputField()
    allowed_transformations: list[str] = dspy.InputField()
    generated_variants: list[str] = dspy.OutputField()
    invariant_to_test: str = dspy.OutputField()


__all__ = [
    "CalibrationDecision",
    "CanonicalizeRepresentation",
    "CaseV3Classifier",
    "ClassifyEquivalenceClass",
    "ClassifyReasoningUnits",
    "DetectInvariantsUnderTransformation",
    "DetectSymmetryBreak",
    "DetermineGrounding",
    "EstimateUncertainty",
    "FrameTask",
    "GenerateTransformationOrbit",
    "MDLControlGate",
    "ScientificMethodCheck",
    "SelectMethod",
    "SelectReasoningUnits",
    "TrackAssumptions",
    "UnitTraceEvaluator",
]
