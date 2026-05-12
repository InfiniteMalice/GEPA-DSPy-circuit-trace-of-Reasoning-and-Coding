# 13-Case Schema V3: Control + Compositional Reasoning Overlay

Schema V3 is an additive overlay on the existing 13+0 abstention,
hallucination, and thought-trace reward schema. It does not replace the base
case identity and does not alter the default confidence threshold `τ = 0.75`.

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
equivalence.
