"""DSPy V3 research overlays on the canonical Mindfulness V5 behavioral contract."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any, Literal

from rg_tracer.abstention.reward_scheme import evaluate_abstention_reward
from rg_tracer.epistemic_cases import (
    FALLBACK_KEY,
    FRAMEWORK_VERSION,
    canonical_case_ids,
    contract_provenance,
    evaluation_identity,
    get_case_key,
    resolve_legacy_case_name,
    validate_coordinate,
)

OutputMode = Literal["answer", "idk", "clarify", "fallback"]
AmbiguityHandlingMode = Literal["answer", "assumptive_proceed", "clarify", "epistemic_abstain"]
ConfidenceBand = Literal["high", "low", "unknown"]
ObservabilityTier = Literal["O0", "O1", "O2", "O3", "O4", "O5"]
ClaimStrength = Literal["none", "weak", "moderate", "strong", "overclaimed"]
ClosureStatus = Literal["closed", "not_closed", "unknown"]

ORIGINAL_CASE_IDS = tuple(range(1, 14))
APPENDED_AMBIGUITY_CASES = {i: get_case_key(i) for i in canonical_case_ids() if i >= 14}
# Compatibility exports are derived views, never independent canonical tables.
CASE_NAMES = {0: FALLBACK_KEY, **{i: get_case_key(i) for i in canonical_case_ids()}}


@dataclass
class ObservabilityOverlay:
    """V2 observability/factuality metadata preserved as a V3 overlay field."""

    tier: ObservabilityTier = "O0"
    has_external_evidence: bool = False
    has_provenance: bool = False
    has_trace_package: bool = False
    has_mech_interp_package: bool = False
    verification_route: str | None = None


@dataclass
class ReasoningOverlay:
    """Public compositional-reasoning unit diagnostics."""

    required_units: list[str] = field(default_factory=list)
    observed_units: list[str] = field(default_factory=list)
    missing_units: list[str] = field(default_factory=list)
    failed_units: list[str] = field(default_factory=list)
    composition_depth: int = 0
    composition_graph: list[dict[str, Any]] | None = None

    def finalize(self) -> None:
        """Populate missing units from required and observed units."""

        observed = set(self.observed_units)
        self.missing_units = [unit for unit in self.required_units if unit not in observed]


@dataclass
class ControlOverlay:
    """Public metacognitive-control diagnostics."""

    required_controls: list[str] = field(default_factory=list)
    observed_controls: list[str] = field(default_factory=list)
    missing_controls: list[str] = field(default_factory=list)
    failed_controls: list[str] = field(default_factory=list)
    answer_mode_decision: str = "unknown"
    grounding_status: str = "unknown"
    calibration_status: str = "unknown"
    method_selection_status: str = "unknown"

    def finalize(self) -> None:
        """Populate missing controls from required and observed controls."""

        observed = set(self.observed_controls)
        self.missing_controls = [ctrl for ctrl in self.required_controls if ctrl not in observed]


@dataclass
class CausalScientificOverlay:
    """Causal and scientific-method diagnostics."""

    causal_types: list[str] = field(default_factory=list)
    scientific_controls: list[str] = field(default_factory=list)
    confounders_considered: list[str] = field(default_factory=list)
    falsification_conditions: list[str] = field(default_factory=list)
    alternative_hypotheses: list[str] = field(default_factory=list)
    causal_claim_strength: ClaimStrength = "none"


@dataclass
class GroupTheoreticOverlay:
    """Transformation, symmetry, invariant, and equivalence diagnostics."""

    transformations: list[str] = field(default_factory=list)
    invariant_properties: list[str] = field(default_factory=list)
    changed_properties: list[str] = field(default_factory=list)
    equivalence_class: str | None = None
    canonical_form: dict[str, Any] | None = None
    symmetry_breaks: list[str] = field(default_factory=list)
    inverse_operations: list[str] = field(default_factory=list)
    closure_status: ClosureStatus | None = None
    orbit_variants: list[str] = field(default_factory=list)
    stabilizer_transformations: list[str] = field(default_factory=list)
    quotient_structure: dict[str, Any] | None = None


@dataclass
class MDLControlOverlay:
    """MDL-control gate diagnostics comparing default and controlled paths."""

    default_answer: str | None = None
    controlled_answer: str | None = None
    default_control_conflict: bool = False
    escalation_required: bool = False
    escalation_taken: bool = False
    compression_candidate: bool = False
    compression_guardrails: list[str] = field(default_factory=list)


@dataclass
class TrajectoryOverlay:
    """Optional public recursive-refinement trajectory diagnostics."""

    sampler_name: str = "trm"
    total_updates: int = 0
    max_depth: int = 0
    max_width: int = 0
    branch_count: int = 0
    prune_count: int = 0
    convergence_detected: bool = False
    budget_exhausted: bool = False
    active_views: list[str] = field(default_factory=list)
    view_route: list[str] = field(default_factory=list)
    revisited_views: list[str] = field(default_factory=list)
    process_score_components: dict[str, float] = field(default_factory=dict)


@dataclass
class LatticeDeductionOverlay:
    """Optional public explicit-lattice deduction diagnostics."""

    mode: str = "off"
    adapter_name: str | None = None
    abstraction_source: str = "explicit"
    initial_candidate_count: int = 0
    remaining_candidate_count: int = 0
    meet_count: int = 0
    join_count: int = 0
    projection_count: int = 0
    canonicalization_count: int = 0
    merged_branch_count: int = 0
    contradiction_detected: bool = False
    resolved: bool = False
    unresolved: bool = False
    abstain_recommended: bool = False
    abstain_reason: str | None = None


@dataclass
class SemanticConstraintOverlay:
    """Optional bounded semantic-constraint compiler diagnostics."""

    mode: str = "off"
    compiler_name: str | None = None
    compiled_constraint_count: int = 0
    verified_constraint_count: int = 0
    unsupported_fragments: list[str] = field(default_factory=list)
    ambiguity_detected: bool = False
    contradiction_detected: bool = False
    provenance_complete: bool = False
    safe_for_shadow_projection: bool = False
    safe_for_gated_projection: bool = False


@dataclass
class RewardComponents:
    """Decomposed V3 reward components."""

    r_token: float = 0.0
    r_confidence: float = 0.0
    r_thought: float = 0.0
    r_abstain: float = 0.0
    r_grounding: float = 0.0
    r_control: float = 0.0
    r_reasoning_unit: float = 0.0
    r_observability: float = 0.0
    r_group_theoretic: float = 0.0
    total: float = 0.0

    def finalize(self) -> None:
        """Recompute total while preserving decomposed component names."""

        self.r_thought = max(0.0, self.r_thought)
        self.total = float(
            self.r_token
            + self.r_confidence
            + self.r_thought
            + self.r_abstain
            + self.r_grounding
            + self.r_control
            + self.r_reasoning_unit
            + self.r_observability
            + self.r_group_theoretic
        )


@dataclass
class Diagnostics:
    """Public V3 diagnostics, excluding private hidden chain-of-thought."""

    primary_failure_mode: str | None = None
    secondary_failure_modes: list[str] = field(default_factory=list)
    over_refusal_risk: float | None = None
    hallucination_risk: float | None = None
    abstention_quality: str | None = None
    ambiguity_handling_score: float | None = None
    repair_recommendation: str | None = None


@dataclass
class CaseV3Result:
    """V5 behavioral result retaining the DSPy V3 overlay API."""

    case_id: int
    base_case_name: str
    output_mode: OutputMode
    is_correct: bool | None
    confidence: float | None
    confidence_band: ConfidenceBand
    threshold_tau: float
    thought_aligned: bool
    hidden_answer_supported: bool | None
    observability: ObservabilityOverlay = field(default_factory=ObservabilityOverlay)
    reasoning_overlay: ReasoningOverlay = field(default_factory=ReasoningOverlay)
    control_overlay: ControlOverlay = field(default_factory=ControlOverlay)
    causal_scientific_overlay: CausalScientificOverlay = field(
        default_factory=CausalScientificOverlay
    )
    group_theoretic_overlay: GroupTheoreticOverlay = field(default_factory=GroupTheoreticOverlay)
    mdl_control_overlay: MDLControlOverlay = field(default_factory=MDLControlOverlay)
    trajectory_overlay: TrajectoryOverlay = field(default_factory=TrajectoryOverlay)
    lattice_deduction_overlay: LatticeDeductionOverlay = field(
        default_factory=LatticeDeductionOverlay
    )
    semantic_constraint_overlay: SemanticConstraintOverlay = field(
        default_factory=SemanticConstraintOverlay
    )
    reward_components: RewardComponents = field(default_factory=RewardComponents)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    compact_label: str = ""
    framework_version: str = FRAMEWORK_VERSION
    stripe: str = "NONE"
    stripe_subtype: str | None = None
    repeat_id: int = 0
    canonical_case_id: int | None = field(init=False)
    canonical_case_key: str | None = field(init=False)
    canonical_case_title: str | None = field(init=False)
    canonical: bool = field(init=False)

    def __post_init__(self) -> None:
        """Validate coordinates, normalize legacy names and populate canonical identity."""
        self.validate_identity()
        self.base_case_name = resolve_legacy_case_name(self.base_case_name)
        identity = evaluation_identity(
            self.case_id, self.stripe, self.stripe_subtype, self.repeat_id
        )
        for key in ("canonical_case_id", "canonical_case_key", "canonical_case_title", "canonical"):
            setattr(self, key, identity[key])

    def validate_identity(self) -> None:
        """Reject invalid coordinates, framework versions or contradictory identity fields."""
        validate_coordinate(self.case_id, self.stripe, self.stripe_subtype, self.repeat_id)
        if self.framework_version != FRAMEWORK_VERSION:
            raise ValueError("CaseV3Result requires framework_version='17case-v5'")
        expected_key = FALLBACK_KEY if self.case_id == 0 else get_case_key(self.case_id)
        if resolve_legacy_case_name(self.base_case_name) != expected_key:
            raise ValueError("base_case_name does not match case_id")
        identity = evaluation_identity(
            self.case_id, self.stripe, self.stripe_subtype, self.repeat_id
        )
        for name in (
            "canonical_case_id",
            "canonical_case_key",
            "canonical_case_title",
            "canonical",
        ):
            if hasattr(self, name):
                value = getattr(self, name)
                if type(value) is not type(identity[name]) or value != identity[name]:
                    raise ValueError(f"{name} contradicts case_id")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        self.validate_identity()
        payload = asdict(self)
        payload.update(
            evaluation_identity(self.case_id, self.stripe, self.stripe_subtype, self.repeat_id)
        )
        payload["base_case_name"] = resolve_legacy_case_name(self.base_case_name)
        payload["contract_provenance"] = contract_provenance()
        payload["overlay_version"] = "v3"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CaseV3Result:
        """Load old/new records, normalizing legacy names and preserving nested overlays."""
        data = dict(payload)
        provenance = data.pop("contract_provenance", None)
        if "contract_provenance" in payload and provenance != contract_provenance():
            raise ValueError("Serialized contract provenance differs from the pinned V5 mirror")
        if data.pop("overlay_version", "v3") != "v3":
            raise ValueError("Unsupported overlay_version")
        identity = evaluation_identity(
            data["case_id"],
            data.get("stripe", "NONE"),
            data.get("stripe_subtype"),
            data.get("repeat_id", 0),
        )
        for name in (
            "canonical_case_id",
            "canonical_case_key",
            "canonical_case_title",
            "canonical",
        ):
            if name in data:
                value = data.pop(name)
                if type(value) is not type(identity[name]) or value != identity[name]:
                    raise ValueError(f"Serialized {name} contradicts case_id")
        overlays = {
            "observability": ObservabilityOverlay,
            "reasoning_overlay": ReasoningOverlay,
            "control_overlay": ControlOverlay,
            "causal_scientific_overlay": CausalScientificOverlay,
            "group_theoretic_overlay": GroupTheoreticOverlay,
            "mdl_control_overlay": MDLControlOverlay,
            "trajectory_overlay": TrajectoryOverlay,
            "lattice_deduction_overlay": LatticeDeductionOverlay,
            "semantic_constraint_overlay": SemanticConstraintOverlay,
            "reward_components": RewardComponents,
            "diagnostics": Diagnostics,
        }
        for name, overlay_type in overlays.items():
            if isinstance(data.get(name), dict):
                data[name] = overlay_type(**data[name])
        return cls(**data)


def compact_label_for(result: CaseV3Result) -> str:
    """Generate a deterministic compact V3 label."""

    tags = [f"Case{result.case_id}", result.observability.tier]
    if result.control_overlay.calibration_status in {"calibrated", "push_down", "push_up"}:
        tags.append("CAL")
    if result.control_overlay.grounding_status in {"grounded", "missing_evidence"}:
        tags.append("GRD")
    if result.causal_scientific_overlay.scientific_controls:
        tags.append("SCI")
    if result.reasoning_overlay.required_units:
        tags.append("RU:" + "+".join(sorted(result.reasoning_overlay.required_units)))
    if result.control_overlay.required_controls:
        tags.append("CTRL:" + "+".join(sorted(result.control_overlay.required_controls)))
    return "-".join(tags)


def _validate_probability(value: float | None, parameter_name: str, *, allow_none: bool) -> None:
    """Reject non-real, boolean or out-of-range probabilities, optionally allowing None."""
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{parameter_name} must be between 0 and 1")
    if isinstance(value, bool) or not isinstance(value, Real):
        if allow_none:
            raise ValueError(f"{parameter_name} must be between 0 and 1 or None")
        raise ValueError(f"{parameter_name} must be between 0 and 1")
    if math.isnan(float(value)) or value < 0.0 or value > 1.0:
        if allow_none:
            raise ValueError(f"{parameter_name} must be between 0 and 1 or None")
        raise ValueError(f"{parameter_name} must be between 0 and 1")


def _confidence_band(confidence: float | None, threshold_tau: float) -> ConfidenceBand:
    """Validate probabilities and return high/low, or unknown for missing confidence."""
    _validate_probability(threshold_tau, "threshold_tau", allow_none=False)
    _validate_probability(confidence, "confidence", allow_none=True)
    if confidence is None:
        return "unknown"
    return "high" if confidence >= threshold_tau else "low"


def _normalize_ambiguity_mode(
    ambiguity_mode: AmbiguityHandlingMode | str | None,
) -> AmbiguityHandlingMode | None:
    """Return an accepted ambiguity mode or reject an unsupported value."""
    if ambiguity_mode is None:
        return None
    valid_modes = ("answer", "assumptive_proceed", "clarify", "epistemic_abstain")
    if ambiguity_mode not in valid_modes:
        raise ValueError(f"ambiguity_mode must be one of {valid_modes}")
    return ambiguity_mode


def _validate_bool_flag(value: bool, parameter_name: str) -> None:
    """Require an actual boolean rather than a truthy substitute."""
    if type(value) is not bool:
        raise TypeError(f"{parameter_name} must be a boolean")


def _score_ambiguity_handling(
    *,
    mode: AmbiguityHandlingMode,
    high_stakes: bool,
    targeted_clarification: bool,
    guessed_silently: bool,
    excessive_questions: bool,
    resumed_after_clarification: bool,
    stalled_after_clarification: bool,
) -> float:
    """Return a 0-4 GEPA-style score for ambiguity handling behavior."""

    if stalled_after_clarification:
        return 1.0 if high_stakes else 1.5
    if resumed_after_clarification and mode == "clarify" and targeted_clarification:
        return 4.0 if high_stakes else 3.0
    if excessive_questions:
        return 1.0 if not high_stakes else 2.0
    if guessed_silently:
        return 1.0 if high_stakes else 2.0
    if mode == "clarify":
        if high_stakes:
            return 3.5 if targeted_clarification else 2.0
        return 1.5 if not targeted_clarification else 2.0
    if mode == "assumptive_proceed":
        return 2.0 if high_stakes else 3.5
    if mode == "answer":
        return 1.5 if high_stakes else 3.0
    return 2.0


def _classify_ambiguity_case(
    *,
    ambiguity_mode: AmbiguityHandlingMode | str | None,
    ambiguity_high_stakes: bool | None,
    targeted_clarification: bool,
    guessed_silently: bool,
    excessive_questions: bool,
    resumed_after_clarification: bool,
    stalled_after_clarification: bool,
) -> tuple[int | None, float] | None:
    """Validate ambiguity flags and return an appended case and score when applicable.

    Low-stakes proceeding keeps the base case with a handling score. Absent mode
    and epistemic abstention leave both base classification and scoring unchanged.
    """
    mode = _normalize_ambiguity_mode(ambiguity_mode)
    for flag_name, flag_value in (
        ("targeted_clarification", targeted_clarification),
        ("guessed_silently", guessed_silently),
        ("excessive_questions", excessive_questions),
        ("resumed_after_clarification", resumed_after_clarification),
        ("stalled_after_clarification", stalled_after_clarification),
    ):
        _validate_bool_flag(flag_value, flag_name)
    if mode is None or mode == "epistemic_abstain":
        return None
    if ambiguity_high_stakes is None:
        raise ValueError(
            "ambiguity_high_stakes must be provided when ambiguity_mode is set "
            "outside epistemic_abstain"
        )
    _validate_bool_flag(ambiguity_high_stakes, "ambiguity_high_stakes")
    high_stakes = ambiguity_high_stakes
    score = _score_ambiguity_handling(
        mode=mode,
        high_stakes=high_stakes,
        targeted_clarification=targeted_clarification,
        guessed_silently=guessed_silently,
        excessive_questions=excessive_questions,
        resumed_after_clarification=resumed_after_clarification,
        stalled_after_clarification=stalled_after_clarification,
    )

    if stalled_after_clarification or excessive_questions:
        return 17, score
    if mode in {"answer", "assumptive_proceed"}:
        return (15 if high_stakes else None), score
    if mode == "clarify" and high_stakes and targeted_clarification:
        return 14, score
    if mode == "clarify" and high_stakes:
        return 17, score
    if mode == "clarify" and not high_stakes:
        return 16, score
    raise ValueError(
        "unsupported ambiguity_mode/appended-case combination: "
        f"ambiguity_mode={mode!r}, appended case not resolved"
    )


def _output_mode_for_case(case_id: int, is_idk: bool, output_text: str) -> OutputMode:
    """Derive answer, IDK, clarification or empty-fallback mode from the final case."""
    if case_id in {14, 16, 17}:
        return "clarify"
    if case_id == 15:
        return "answer"
    if is_idk:
        return "idk"
    if case_id == 0 and not output_text.strip():
        return "fallback"
    return "answer"


def _base_prediction(output_text: str) -> str:
    """Preserve answer text for the historical abstention reward evaluator."""
    return output_text


def classify_case_v3(
    *,
    output_text: str,
    expected_answer: str | None,
    is_idk: bool,
    confidence: float | None,
    threshold_tau: float = 0.75,
    thought_aligned: bool,
    hidden_answer_supported: bool | None = None,
    observability: ObservabilityOverlay | None = None,
    reasoning_overlay: ReasoningOverlay | None = None,
    control_overlay: ControlOverlay | None = None,
    causal_scientific_overlay: CausalScientificOverlay | None = None,
    group_theoretic_overlay: GroupTheoreticOverlay | None = None,
    mdl_control_overlay: MDLControlOverlay | None = None,
    trajectory_overlay: TrajectoryOverlay | None = None,
    lattice_deduction_overlay: LatticeDeductionOverlay | None = None,
    semantic_constraint_overlay: SemanticConstraintOverlay | None = None,
    ambiguity_mode: AmbiguityHandlingMode | str | None = None,
    ambiguity_high_stakes: bool | None = None,
    targeted_clarification: bool = False,
    guessed_silently: bool = False,
    excessive_questions: bool = False,
    resumed_after_clarification: bool = False,
    stalled_after_clarification: bool = False,
    stripe: str = "NONE",
    stripe_subtype: str | None = None,
    repeat_id: int = 0,
) -> CaseV3Result:
    """Classify V5 behavior independently of stripe/repeat and V3 research overlays.

    Missing confidence forces answer/IDK behavior to Case 0 with neutral rewards.
    Ambiguity Cases 14–17 retain their existing rewards without observed confidence.
    """

    _validate_probability(threshold_tau, "threshold_tau", allow_none=False)
    _validate_probability(confidence, "confidence", allow_none=True)

    obs = observability or ObservabilityOverlay()
    reasoning = reasoning_overlay or ReasoningOverlay()
    control = control_overlay or ControlOverlay()
    causal = causal_scientific_overlay or CausalScientificOverlay()
    group = group_theoretic_overlay or GroupTheoreticOverlay()
    mdl = mdl_control_overlay or MDLControlOverlay()
    trajectory = trajectory_overlay or TrajectoryOverlay()
    lattice = lattice_deduction_overlay or LatticeDeductionOverlay()
    semantic_constraints = semantic_constraint_overlay or SemanticConstraintOverlay()
    reasoning.finalize()
    control.finalize()

    eval_confidence = 0.0 if confidence is None else confidence
    if is_idk:
        prediction = expected_answer if hidden_answer_supported else "I don't know"
    else:
        prediction = _base_prediction(output_text)
    outcome = evaluate_abstention_reward(
        expected_answer=expected_answer,
        prediction=prediction,
        text=output_text,
        confidence=eval_confidence,
        aligned=thought_aligned,
        abstained=is_idk,
        config={"abstention": {"threshold": threshold_tau}},
    )
    ambiguity_case = _classify_ambiguity_case(
        ambiguity_mode=ambiguity_mode,
        ambiguity_high_stakes=ambiguity_high_stakes,
        targeted_clarification=targeted_clarification,
        guessed_silently=guessed_silently,
        excessive_questions=excessive_questions,
        resumed_after_clarification=resumed_after_clarification,
        stalled_after_clarification=stalled_after_clarification,
    )
    # Unknown confidence cannot establish either of V5's observed confidence bands.
    # Ambiguity cases below have not_applicable confidence semantics and remain eligible.
    case_id = outcome.case_id if confidence is not None else 0
    ambiguity_handling_score = None
    if ambiguity_case is not None:
        ambiguity_id, ambiguity_handling_score = ambiguity_case
        if ambiguity_id is not None:
            case_id = ambiguity_id

    rewards = RewardComponents(
        r_token=outcome.components.get("token", 0.0),
        r_confidence=outcome.components.get("confidence", 0.0),
        r_thought=outcome.components.get("thought", 0.0),
        r_abstain=outcome.components.get("abstain", 0.0),
        r_grounding=_score_grounding(obs, control),
        r_control=_score_control(control, mdl),
        r_reasoning_unit=_score_reasoning_units(reasoning),
        r_observability=_score_observability(obs),
        r_group_theoretic=_score_group_theoretic(group),
    )
    rewards.finalize()
    if confidence is None and case_id == 0:
        # No observed band supports a reward, including bonuses from diagnostic overlays.
        rewards = RewardComponents()

    result = CaseV3Result(
        case_id=case_id,
        base_case_name=CASE_NAMES[case_id],
        output_mode=_output_mode_for_case(case_id, is_idk, output_text),
        is_correct=None if case_id >= 14 else outcome.correct,
        confidence=confidence,
        confidence_band=_confidence_band(confidence, threshold_tau),
        threshold_tau=threshold_tau,
        thought_aligned=thought_aligned,
        hidden_answer_supported=hidden_answer_supported,
        stripe=stripe,
        stripe_subtype=stripe_subtype,
        repeat_id=repeat_id,
        observability=obs,
        reasoning_overlay=reasoning,
        control_overlay=control,
        causal_scientific_overlay=causal,
        group_theoretic_overlay=group,
        mdl_control_overlay=mdl,
        trajectory_overlay=trajectory,
        lattice_deduction_overlay=lattice,
        semantic_constraint_overlay=semantic_constraints,
        reward_components=rewards,
        diagnostics=_diagnose(case_id, control, causal, group, mdl, ambiguity_handling_score),
    )
    result.compact_label = compact_label_for(result)
    return result


def _score_grounding(obs: ObservabilityOverlay, control: ControlOverlay) -> float:
    """Award the existing grounding bonus for evidence, provenance or grounded control."""
    grounded = (
        obs.has_external_evidence or obs.has_provenance or control.grounding_status == "grounded"
    )
    return 0.25 if grounded else 0.0


def _score_control(control: ControlOverlay, mdl: MDLControlOverlay) -> float:
    """Sum matched required controls and a completed required escalation bonus."""
    required = set(control.required_controls)
    observed = set(control.observed_controls)
    score = 0.0 if not required else 0.1 * len(required & observed)
    if mdl.escalation_required and mdl.escalation_taken:
        score += 0.25
    return score


def _score_reasoning_units(reasoning: ReasoningOverlay) -> float:
    """Score matched required units and successful multi-step composition."""
    required = set(reasoning.required_units)
    observed = set(reasoning.observed_units)
    if not required:
        return 0.0
    composed_bonus = 0.1 if reasoning.composition_depth > 1 and not reasoning.failed_units else 0.0
    return 0.1 * len(required & observed) + composed_bonus


def _score_observability(obs: ObservabilityOverlay) -> float:
    """Map the observability tier to its existing diagnostic reward bonus."""
    tier_scores = {"O0": 0.0, "O1": 0.05, "O2": 0.1, "O3": 0.2, "O4": 0.3, "O5": 0.4}
    return tier_scores.get(obs.tier, 0.0)


def _score_group_theoretic(group: GroupTheoreticOverlay) -> float:
    """Award one bonus for each populated group-theoretic diagnostic category."""
    signals = [
        bool(group.invariant_properties),
        bool(group.equivalence_class),
        bool(group.canonical_form),
        bool(group.symmetry_breaks),
        bool(group.inverse_operations),
        bool(group.orbit_variants),
    ]
    return 0.1 * sum(signals)


def _diagnose(
    case_id: int,
    control: ControlOverlay,
    causal: CausalScientificOverlay,
    group: GroupTheoreticOverlay,
    mdl: MDLControlOverlay,
    ambiguity_handling_score: float | None,
) -> Diagnostics:
    """Collect public failure, abstention and repair diagnostics for the final case."""
    failures = list(control.failed_controls)
    if causal.causal_claim_strength == "overclaimed":
        failures.append("causal_overclaim")
    if "false_equivalence" in group.changed_properties:
        failures.append("false_equivalence")
    if mdl.escalation_required and not mdl.escalation_taken:
        failures.append("missed_escalation")
    primary = failures[0] if failures else None
    abstention_quality = "lazy" if case_id == 9 else None
    if case_id == 12:
        abstention_quality = "grounded_uncertainty"
    if case_id == 14:
        abstention_quality = "high_stakes_ambiguity_clarification"
    elif case_id == 16:
        abstention_quality = "over_clarification"
    elif case_id == 17:
        abstention_quality = "clarification_loop_or_stall"
    return Diagnostics(
        primary_failure_mode=primary,
        secondary_failure_modes=failures[1:],
        over_refusal_risk=0.8 if control.answer_mode_decision == "blanket_refusal" else None,
        hallucination_risk=0.9 if case_id == 6 else None,
        abstention_quality=abstention_quality,
        ambiguity_handling_score=ambiguity_handling_score,
        repair_recommendation="escalate_control" if primary == "missed_escalation" else None,
    )


__all__ = [
    "APPENDED_AMBIGUITY_CASES",
    "CASE_NAMES",
    "ORIGINAL_CASE_IDS",
    "AmbiguityHandlingMode",
    "CaseV3Result",
    "CausalScientificOverlay",
    "ControlOverlay",
    "Diagnostics",
    "GroupTheoreticOverlay",
    "LatticeDeductionOverlay",
    "MDLControlOverlay",
    "ObservabilityOverlay",
    "ReasoningOverlay",
    "RewardComponents",
    "SemanticConstraintOverlay",
    "TrajectoryOverlay",
    "classify_case_v3",
    "compact_label_for",
]
