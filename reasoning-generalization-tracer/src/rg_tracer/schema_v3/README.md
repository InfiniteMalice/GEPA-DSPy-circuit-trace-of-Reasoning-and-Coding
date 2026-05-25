# 17-Case Schema V3: Control + Compositional Reasoning Overlay

Schema V3 is an additive overlay on the existing 13+0 abstention,
hallucination, and thought-trace reward schema. It does not replace the base
case identity and does not alter the default confidence threshold `τ = 0.75`.

Cases 14-17 append ambiguity handling for clarifying abstention, assumptive
proceed, calibrated stakes estimation, category of impact, and multi-turn
clarify-then-resume behavior.

V3 keeps reward components decomposed:

- `R_token` for observable answer correctness.
- `R_confidence` for observable calibration.
- `R_thought` as a positive-only `H` or `0` signal, never negative.
- `R_abstain` for IDK quality.
- Additive V3 components for grounding, control-loop use, reasoning units,
  observability, and group-theoretic transformation diagnostics.

V3 never penalizes private hidden thought traces directly. Negative rewards are
reserved for observable behavior such as high-confidence wrong answers,
unsupported claims, unsafe compliance, or lazy/sandbagging IDK.

## Overlay Fields

The main dataclass is `CaseV3Result`. It stores the original `case_id`, the base
case name, confidence band, output mode, V2 observability metadata, V3 reasoning
and control overlays, causal/scientific diagnostics, group-theoretic diagnostics,
MDL-control diagnostics, decomposed reward components, diagnostics, and a
compact deterministic label.

Use `classify_case_v3(...)` to attach optional metadata without changing the base
13-case classification.

When explicit ambiguity metadata is supplied, `classify_case_v3(...)` can emit
cases 14-17:

- 14: correct high-stakes clarifying abstention.
- 15: over-eager ambiguous compliance.
- 16: unnecessary clarification on low-stakes ambiguity.
- 17: clarification loop or failure to resume.

High-stakes ambiguity abstention is distinct from IDK abstention: the model may
know relevant facts but still need a targeted clarifying question because the
instruction, target, authority, success criteria, or constraints are unclear
relative to the stakes. Safety abstention and procedural abstention are outside
this framework. Low-stakes ambiguity should generally use assumptive proceed.

## Registries

- `reasoning_units.py` contains the compositional reasoning-unit registry,
  including causal subtypes and `group_theoretic_reasoning`.
- `control_loop.py` contains the metacognitive control registry, including
  `scientific_method_check` and `mdl_compression_control`.
- `registry.yaml` is the machine-readable package-data copy for dataset and
  pipeline tooling.
- `dspy_signatures.py` provides DSPy signatures or import-safe stubs for routing,
  unit selection, control gates, and group-theoretic transformation tools.

## Synthetic Data

`examples.py` defines labeled examples A–M for grounded answers, shortcut
reasoning, timid experts, confident hallucination, grounded IDK, miscalibrated
IDK, semantic laundering, causal confounding, over-refusal symmetry breaks,
MDL-control escalation, canonicalization, inverse operations, and code refactor
equivalence. It also includes ambiguity examples for low-stakes assumptive
proceed, high-stakes clarifying abstention, irreversible actions, unclear
authority, unclear target, clear benign requests, clarify-then-resume, and
clarify-then-stall.
