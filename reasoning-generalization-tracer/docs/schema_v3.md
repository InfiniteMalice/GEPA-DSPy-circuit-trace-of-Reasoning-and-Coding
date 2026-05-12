# 13-Case Schema V3: Control + Compositional Reasoning Overlay

Schema V3 extends the existing 13+0 abstention / hallucination / thought-trace
reward schema with public control-loop and compositional-reasoning diagnostics.
It is an overlay, not a replacement: cases 0–13 keep their original meanings,
and `docs/epistemic_alignment.md` remains the V1/V2 reward foundation.

## V1 / V2 / V3 Relationship

- **V1 behavioral case identity:** answer versus IDK, correctness, confidence,
  and thought alignment.
- **V2 observability and factuality overlay:** verification tier O0–O5,
  evidence, provenance, trace packages, repair routes, semantic robustness, and
  certification metadata.
- **V3 control + compositional reasoning overlay:** reasoning-unit requirements,
  control operations, causal/scientific checks, MDL-control gates, and
  group-theoretic transformation diagnostics.

V3 does not introduce negative hidden-thought penalties. `R_thought` remains
positive-only (`H` or `0`), and hidden/internal thought traces are never directly
penalized. Negative reward remains tied to observable final-answer behavior,
confidence miscalibration, unsupported claims, unsafe compliance, or
lazy/sandbagging IDK.

V3 separates answer correctness, verification quality, reasoning-unit use,
control-loop quality, and transformation-stability diagnostics. It is compatible
with GEPA, GRPO, PPO+GRN, DAPO-hybrid, DSPy pipelines, circuit tracing,
attribution graphs, semantic intent robustness, and factuality certification.

## Data Model

The implementation lives in `src/rg_tracer/schema_v3/` and uses stdlib
`dataclasses`:

- `CaseV3Result` preserves `case_id`, `base_case_name`, output mode, correctness,
  confidence, confidence band, `threshold_tau`, thought alignment, and optional
  hidden-answer support.
- `ObservabilityOverlay` records O0–O5, evidence, provenance, trace packages,
  mechanistic-interpretability packages, and verification routes.
- `ReasoningOverlay` records required, observed, missing, and failed reasoning
  units plus composition depth and optional composition graph.
- `ControlOverlay` records required, observed, missing, and failed controls plus
  answer-mode, grounding, calibration, and method-selection status.
- `CausalScientificOverlay` records causal types, scientific controls,
  confounders, falsification conditions, alternative hypotheses, and causal claim
  strength.
- `GroupTheoreticOverlay` records transformations, invariants, changed
  properties, equivalence class, canonical form, symmetry breaks, inverse
  operations, closure status, orbit variants, stabilizers, and quotient
  structure.
- `MDLControlOverlay` records default and controlled answers, conflict signals,
  escalation decisions, compression candidates, and guardrails.

`classify_case_v3(...)` accepts the existing case-classification inputs plus
optional V3 overlays and returns a JSON-serializable result with decomposed
reward components and a deterministic compact label.

## Reward Logic

V3 augments, rather than rewrites, the base reward scheme:

- `r_grounding` is positive when evidence, provenance, or grounded control status
  is present.
- `r_control` is positive when required control operations are observed and when
  MDL escalation is correctly taken.
- `r_reasoning_unit` is positive when required reasoning units are present and
  correctly composed.
- `r_observability` is positive for useful O1–O5 verification metadata when
  available or required.
- `r_group_theoretic` is positive when relevant invariants, equivalence classes,
  canonical forms, symmetry breaks, inverse operations, or orbits are identified.

These components are additive and decomposed. They do not collapse the schema
into a monolithic reward and do not force over-refusal.

## Reasoning Unit Registry

`reasoning_units.py` and `registry.yaml` define the V3 reasoning families:

1. recursive
2. functional_composition
3. type_constrained_composition
4. abstraction
5. instantiation
6. variable_binding
7. relational_composition
8. constraint_composition
9. decomposition
10. proof_step_composition
11. analogy
12. invariant_preservation
13. contextual_modulation
14. causal_reasoning
15. hierarchical_composition
16. dialectical_composition
17. compression_expansion
18. group_theoretic_reasoning

Causal reasoning is a family with subtypes such as temporal order, mechanism,
counterfactual dependence, interventionist reasoning, common-cause confounding,
necessary-versus-sufficient conditions, probabilistic causality, dose response,
level shifting, agentic causality, and normative responsibility.

## Control-Loop Registry

`control_loop.py` and `registry.yaml` define public control operations:

task framing, epistemic grounding, method selection, reasoning-unit selection,
assumption tracking, uncertainty estimation, calibration decision, consistency
checking, error localization, revision control, scientific method check,
epistemic-boundary abstention, and MDL compression control.

`scientific_method_check` includes hypothesis formation, operationalization,
prediction, falsification conditions, control groups, confounder detection,
measurement validity, replication checks, effect-size reasoning, and alternative
hypothesis comparison.

`mdl_compression_control` compares a fast/default answer with a
controlled/deliberative answer, emits a conflict signal, applies an escalation
rule, identifies safe compression candidates, and records guardrails against
unsafe overcompression.

## Group-Theoretic Reasoning: Symmetry, Invariance, and Equivalence

Group theory is used as a practical reasoning lens, not as a requirement that
every problem be formal algebra. It helps identify when surface transformations
preserve or change relevant structure.

It strengthens semantic laundering detection by treating paraphrases,
translations, wrappers, encodings, and multi-turn fragments as transformations
over an underlying intent representation. If the dangerous capability or causal
affordance is invariant, the variants can be grouped into the same equivalence
class even when the surface wording changes.

It also strengthens over-refusal prevention by detecting symmetry breaks: same
topic does not mean same intent, same action does not mean same authorization,
and same words do not mean same risk. A defensive, educational, or authorized
request can deserve scoped safe help even when a superficially similar malicious
request should be refused or routed.

For mechanistic interpretability, this gives a vocabulary for invariant circuits,
equivalent behaviors, canonical forms, transformation-stable concepts,
stabilizers, orbits, quotient structures, and symmetry-breaking features.

Conceptual framing:

- **Category theory:** how reasoning transformations compose across typed
  structures.
- **Lambda calculus:** how operations bind variables and compute through
  substitution/application.
- **Causal reasoning:** how interventions, mechanisms, and consequences
  propagate.
- **Group theory:** which transformations preserve structure, which break
  symmetry, and which variants belong to the same equivalence class.

Group-theoretic subtypes include identity transformation, inverse
transformation, closure under operation, associativity of composition, symmetry
detection, invariance under transformation, equivalence-class reasoning, orbit
reasoning, stabilizer reasoning, quotient-structure reasoning, normal-form
reasoning, isomorphism detection, symmetry breaking, and conservation-law
analogy.

## Synthetic Examples

The V3 examples cover:

- Correct high-confidence grounded answers.
- Correct answers with shortcut or unfaithful reasoning.
- Timid experts.
- Confident hallucinations.
- Honest grounded IDK.
- Miscalibrated grounded IDK.
- Semantic laundering as invariant preservation and equivalence-class reasoning.
- Causal confounding.
- Over-refusal guards through symmetry breaking.
- MDL-control gates.
- Group-theoretic canonicalization.
- Group-theoretic inverse operations.
- Code refactor equivalence.

At least five examples use group-theoretic reasoning: semantic laundering,
over-refusal symmetry breaking, canonicalization, inverse operations, and code
refactor equivalence.

## DSPy / Pipeline Integration

`dspy_signatures.py` provides lightweight signatures or safe stubs for:

- `ClassifyReasoningUnits`
- `SelectReasoningUnits`
- `FrameTask`
- `DetermineGrounding`
- `SelectMethod`
- `TrackAssumptions`
- `EstimateUncertainty`
- `CalibrationDecision`
- `ScientificMethodCheck`
- `MDLControlGate`
- `UnitTraceEvaluator`
- `CaseV3Classifier`
- `DetectInvariantsUnderTransformation`
- `ClassifyEquivalenceClass`
- `CanonicalizeRepresentation`
- `DetectSymmetryBreak`
- `GenerateTransformationOrbit`

These support synthetic dataset generation, DSPy routing, GEPA scoring, semantic
verification and repair, factuality certification, attribution graphs,
circuit-trace diagnostics, control-loop training, abstention calibration,
compositional reasoning curricula, semantic laundering detection, over-refusal
prevention, and transformation-stability testing.
