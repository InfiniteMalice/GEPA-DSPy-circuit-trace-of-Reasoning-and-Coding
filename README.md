# GEPA DSPy: Circuit Trace of Reasoning and Coding

This repository hosts the **Reasoning Generalisation Tracer (RG-Tracer)**, a
research toolkit for evaluating and improving reasoning systems through
multi-axis scoring, self-play, structured reasoning traces, calibrated
abstention, circuit-level concept rewards, and bounded recursive-refinement
experiments.

RG-Tracer is designed as a lightweight, CPU-friendly scaffold. It is not a
claim that any one experimental architecture has solved general reasoning. The
goal is to make reasoning quality more inspectable, decomposable, testable, and
amenable to iterative improvement.

## How to Use

Clone the repository and install the Python package in editable mode:

```bash
git clone https://github.com/InfiniteMalice/GEPA-DSPy-circuit-trace-of-Reasoning-and-Coding.git
cd GEPA-DSPy-circuit-trace-of-Reasoning-and-Coding
pip install -e reasoning-generalization-tracer
```

Run self-play with the default Tiny Recursion Model (`trm`) sampler:

```bash
rg-tracer self-play \
  --profile proof_math \
  --problem reasoning-generalization-tracer/datasets/toy_math/addition_small.jsonl \
  --concept parity \
  --sampler trm \
  --k 8
```

Run the optional recursive-refinement scaffold:

```bash
rg-tracer self-play \
  --profile proof_math \
  --problem reasoning-generalization-tracer/datasets/toy_math/addition_small.jsonl \
  --concept parity \
  --sampler gram_mdt \
  --k 4 \
  --max-depth 6 \
  --max-width 4 \
  --max-total-updates 48 \
  --seed 7
```

Export a concept trace:

```bash
rg-tracer trace \
  --run runs/<run-id>/best.json \
  --out runs/<run-id>/circuits/ \
  --concept parity
```

Evaluate transfer datasets and write a CSV report:

```bash
rg-tracer eval \
  --dataset "reasoning-generalization-tracer/datasets/transfer_tests/*.jsonl" \
  --profile proof_math \
  --output transfer_eval.csv
```

The package README contains additional commands, configuration examples, and
expected output artifacts:
[`reasoning-generalization-tracer/README.md`](reasoning-generalization-tracer/README.md).

## What RG-Tracer Studies

RG-Tracer combines several complementary approaches:

- **GEPA-style reflective improvement**: evaluate trajectories with decomposed
  feedback rather than reducing all quality judgments to one opaque score.
- **DSPy-compatible pipelines**: structure reasoning tasks as modular,
  optimizable components.
- **Reasoning and coding fitness scoring**: judge whether an answer works, why it
  works, what it omits, and whether the same concepts transfer to new problems.
- **Thought-trace rewards**: reward epistemically aligned reasoning without
  directly punishing hidden thoughts.
- **Calibrated abstention**: distinguish justified uncertainty from lazy refusal,
  hallucination, and irresponsible guessing under ambiguity.
- **Atomic thought decomposition**: break reasoning into inspectable functional
  units that can be scored, recombined, and tested for reuse.
- **Bayesian fallback reasoning**: produce transparent, uncertainty-calibrated
  positions for domains where claims cannot be cleanly verified by a single
  deterministic test.
- **Circuit traces and attribution graphs**: measure whether concepts appear in
  sparse, repeatable, and transferable internal pathways.
- **Experimental recursive refinement**: compare deterministic recursion,
  stochastic widening, explicit deduction projection, and multi-view refinement
  under bounded compute budgets.

The repository is inspired by [DSPy](https://arxiv.org/abs/2310.03714) and
[GEPA](https://arxiv.org/abs/2507.19457), but it extends their use toward a
research environment for reasoning quality, coding quality, abstention,
traceability, and compositional generalization.

## GEPA Concepts for Judging Reasoning and Coding Quality

GEPA is useful here because coding and reasoning quality are not one-dimensional.
A solution can pass a narrow test while still being brittle, unclear,
inefficient, poorly grounded, or incapable of transfer. RG-Tracer therefore
uses deterministic integer scores from **0 to 4** across eleven axes:

| Axis | What it asks about reasoning | What it asks about code |
| --- | --- | --- |
| **Logical Validity** | Do the conclusions follow from the premises? | Does the implementation satisfy the intended behavior without contradictions? |
| **Conceptual Clarity** | Are terms and variables defined consistently? | Are names, interfaces, state transitions, and assumptions understandable? |
| **Completeness** | Are edge cases and boundary conditions covered? | Are failure modes, unusual inputs, and integration paths handled? |
| **Rigor** | Are important steps justified and checked? | Are claims supported by tests, invariants, validation, or reproducible checks? |
| **Efficiency** | Is the path unnecessarily long or wasteful? | Does the implementation avoid avoidable complexity and excessive compute? |
| **Heuristic Creativity** | Does the approach explore useful alternatives? | Does it find robust approaches beyond the first plausible patch? |
| **Numerical Accuracy** | Are calculations tolerance-aware and correct? | Are numerical operations, units, limits, and precision requirements handled correctly? |
| **Cognitive Efficiency** | Does the reasoning stay within token, time, and memory budgets? | Does the solution use an appropriate amount of analysis and tool work? |
| **Explanatory Power** | Does it explain causal links and why the method works? | Can a reviewer understand why the patch is correct and maintain it later? |
| **Self-Consistency** | Does self-critique catch and repair mistakes? | Does the model detect regressions, revise weak code, and preserve earlier constraints? |
| **Abstraction & Generalization** | Does the method transfer beyond the example? | Does the solution use reusable concepts, contracts, helpers, and tests rather than overfit patches? |

Profiles in
[`reasoning-generalization-tracer/src/rg_tracer/scoring/profiles.yaml`](reasoning-generalization-tracer/src/rg_tracer/scoring/profiles.yaml)
weight these axes through a geometric mean with hard gates on logical validity,
rigor, and numerical accuracy.

The point is not to maximize every metric blindly. The point is to expose the
shape of a solution: where it is strong, where it is weak, and whether an
apparently successful answer is relying on shortcuts.

## Value Decomposition: Understanding the Right Concepts

RG-Tracer treats **value decomposition** as a cross-cutting scoring principle.
Before optimizing a response, the system should identify what is actually being
optimized and separate deep requirements from shallow proxies.

For coding and reasoning tasks, useful decomposition includes:

- the user's real objective;
- hard constraints that must not be violated;
- correctness conditions and invariants;
- acceptable tradeoffs among clarity, speed, cost, and maintainability;
- uncertainty that should trigger verification or abstention;
- evidence needed to justify the result;
- low-risk assumptions that may be stated explicitly;
- high-stakes ambiguities that require clarification before acting.

This is intended to reduce proxy gaming. A code generator should not treat
"make the test pass" as equivalent to "solve the underlying problem." A
reasoning model should not treat fluent agreement as equivalent to a grounded
answer. The same decomposition also improves rating: evaluators can score the
specific concepts that mattered rather than reward a superficially convincing
result.

The alignment motivation is related to the
[Deep Value Benchmark](https://arxiv.org/abs/2511.02109), which tests whether
models learn deeper values or merely generalize shallow preferences correlated
with them.

## Thought-Trace Rewards and the 17-Case Schema

All training modes in this repository share a behavioral schema for epistemic
confidence, truthfulness, abstention, and high-stakes ambiguity handling. The
schema preserves the original **13 cases plus a null fallback (`case 0`)**, then
adds four ambiguity-handling cases.

The core reward decomposition remains:

- **`R_token`**: surface-answer correctness;
- **`R_confidence`**: calibration around the default abstention threshold
  `tau = 0.75`;
- **`R_thought`**: a positive-only epistemic-alignment bonus, either `H` or `0`;
- **`R_abstain`**: reward or penalty for choosing `"I don't know"` when it is or
  is not appropriate.

The key invariant is deliberate: **hidden thought traces are never directly
punished.** Aligned reasoning may receive a bonus. Unaligned reasoning receives
no thought bonus. Negative reward remains tied to observable final-answer
behavior, confidence miscalibration, unsupported claims, unsafe compliance, or
lazy/sandbagging abstention.

### Version History

| Version | Scope | What it adds |
| --- | --- | --- |
| **V1: Behavioral cases** | Cases `0-13` | Answer vs. IDK, correctness, confidence, thought alignment, grounded uncertainty, hallucination control, and lazy/sandbagging IDK detection. |
| **V2: Observability and factuality overlay** | Preserves `0-13` | Verification tiers `O0-O5`, evidence, provenance, trace packages, repair routes, semantic robustness, and factuality-certification metadata. |
| **V3: Control and compositional-reasoning overlay** | Preserves `0-13`, appends `14-17` | Clarifying abstention, assumptive proceed, multi-turn ambiguity handling, public reasoning units, control operations, causal/scientific checks, MDL-control gates, group-theoretic diagnostics, and recursive-trajectory metadata. |

### Appended Ambiguity Cases

| Case | Behavior | Meaning |
| --- | --- | --- |
| **14** | Correct high-stakes clarifying abstention | The model pauses and asks the minimum useful clarification because guessing would be irresponsible. |
| **15** | Over-eager ambiguous compliance | The model guesses under unclear high-stakes instructions instead of clarifying. |
| **16** | Unnecessary clarification on low-stakes ambiguity | The model asks when a reversible assumption would have been more helpful. |
| **17** | Clarification loop or failure to resume | The model asks vague or repeated questions, ignores an answer, or stalls instead of completing the task. |

This separates ordinary **IDK abstention** from **high-stakes ambiguity
abstention**. A model may know the relevant facts and still need to pause because
the target, authority, constraints, or success criteria are underspecified.
Safety refusal remains part of the normal safety pipeline and is not added as a
new schema category.

Detailed schema documentation:

- [`reasoning-generalization-tracer/docs/17_case_framework.md`](reasoning-generalization-tracer/docs/17_case_framework.md)
- [`reasoning-generalization-tracer/docs/schema_v3.md`](reasoning-generalization-tracer/docs/schema_v3.md)
- [`reasoning-generalization-tracer/docs/epistemic_alignment.md`](reasoning-generalization-tracer/docs/epistemic_alignment.md)

## Atomic Thoughts and Compositional Reasoning

The **Atomic Thought** idea is that complex reasoning should be decomposed into
fine-grained functional units instead of judged only by its final answer. This
repository adapts that idea for public, structured reasoning metadata and
compositional diagnostics. It does not attempt to reproduce the full
Atom-Searcher training system.

The V3 reasoning-unit registry includes:

1. recursive reasoning;
2. functional composition;
3. type-constrained composition;
4. abstraction;
5. instantiation;
6. variable binding;
7. relational composition;
8. constraint composition;
9. decomposition;
10. proof-step composition;
11. analogy;
12. invariant preservation;
13. contextual modulation;
14. causal reasoning;
15. hierarchical composition;
16. dialectical composition;
17. compression and expansion;
18. group-theoretic reasoning.

Atomic thoughts are useful because they provide inspectable anchors for scoring
and training. Instead of asking only whether a solution succeeded, RG-Tracer can
ask:

- Which reasoning units were required?
- Which units appeared in the public trace?
- Which units were missing, malformed, or poorly composed?
- Did the model preserve invariants under a code refactor or semantic paraphrase?
- Did a concept transfer across related tasks?
- Did recursive refinement improve verification, or merely add redundant steps?

This is related to
[Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward](https://arxiv.org/abs/2508.12800),
which decomposes reasoning into fine-grained functional units and uses atomic
thought rewards for process-level guidance.

## Factuality Certification and Over-Refusal Guard

The optional `factuality_certification` layer is inspired by the broader idea of
constraint-based certification. It is not a formal proof of truth. It certifies
answers relative to the available evidence and context, supports
`off` / `shadow` / `advisory` / `gated` / `training` modes, and distinguishes:

- refusal;
- epistemic abstention;
- scoped partial answers;
- uncertainty-qualified answers;
- grounded full answers.

The over-refusal guard matters because hallucination control should not collapse
into unhelpful refusal. A model should answer confidently when evidence supports
an answer, qualify uncertainty when appropriate, and abstain only when the
remaining uncertainty is material.

This layer is inspired in part by
[GeoCert: Certified Geometric AI for Reliable Forecasting](https://arxiv.org/abs/2604.23474),
while adapting the certification idea conservatively to answer-level evidence
and observability metadata.

## Bayesian Reasoning for Non-Verifiable Domains

Some domains do not provide a clean unit test, formal proof, or single
experiment that can settle the question. Historical interpretation, policy
analysis, ethics, social science, and other humanities problems often require a
position that is evidence-weighted rather than mechanically verified. RG-Tracer
uses a separate **academic-to-Bayesian fallback pipeline** for these cases.

The fallback does not treat a posterior probability as objective truth. It is a
structured way to expose assumptions, compare evidence, calibrate uncertainty,
and identify where a conclusion remains fragile.

The pipeline proceeds in three stages:

1. **Academic rigor evaluation.** The claim is checked against humanities-focused
   criteria such as source handling, interpretive fidelity, historiography,
   causal discipline, triangulation, normative-versus-positive separation,
   uncertainty calibration, intellectual charity, rhetorical hygiene,
   reproducibility, synthesis, and epistemic neutrality.
2. **Semantic verification and abstention.** The verifier flags contradictions,
   unsupported claims, misquotes, context errors, over-claimed causality, and
   is-ought slips. Hard gates remain authoritative. If the analysis fails the
   humanities gates or confidence remains below the calibrated threshold, the
   pipeline abstains instead of forcing a position.
3. **Bayesian position synthesis.** For a claim that passes the gates, the system
   combines an explicit prior with evidence-specific likelihoods and reports the
   posterior, dominant evidence, sensitivity summary, and decision policy.

The current decision policy is deliberately conservative:

| Posterior | Default policy |
| --- | --- |
| `>= 0.75` | Support with caveats |
| `> 0.25` and `< 0.75` | Gather more evidence |
| `<= 0.25` | Recommend caution |

The output is intended to be auditable. Evidence records include the source,
year, method, finding, and limitations. Priors and likelihoods remain explicit
so researchers can inspect whether the conclusion depends too heavily on a
contestable assumption or a single dominant source.

Run the fallback pipeline with:

```bash
rg-tracer fallback \
  --problem reasoning-generalization-tracer/datasets/humanities/sample_claims.jsonl
```

The implementation lives in
[`reasoning-generalization-tracer/src/rg_tracer/fallback/`](reasoning-generalization-tracer/src/rg_tracer/fallback/).
Humanities scoring is documented in the package README:
[`reasoning-generalization-tracer/README.md`](reasoning-generalization-tracer/README.md).

## Experimental Recursive Refinement

The default self-play sampler remains `sampler="trm"`. Recursive refinement is
an additive experimental branch, not a replacement for the baseline.

### `gram_mdt`: Adaptive Multi-View Refinement

The optional `sampler="gram_mdt"` scaffold explores:

- recursive state refinement;
- seeded stochastic widening;
- adaptive depth and bounded branching;
- uncertainty-aware allocation of additional compute;
- public view routing through operators such as arithmetic, parity, constraint
  checking, counterexample probing, verification, and compression;
- decomposed process scores for confidence, uncertainty reduction, verification,
  convergence, efficiency, diversity, redundancy, contradiction counts, and
  budget use.

The controller starts with a shallow pass, then applies public refinement
operators until confidence is high enough, uncertainty is low enough, branches
converge, no useful view remains, or a configured compute budget is exhausted.

Process scores are heuristic and decomposed. Their ranking influence defaults to
`process_reward_weight = 0.0`. Hard gates remain authoritative, and no negative
reward is applied directly to hidden thought traces.

Runs add:

- `trajectories.jsonl`;
- `view_routes.jsonl`;
- `budget_metrics.json`;
- recursive-refinement fields in `scores.jsonl` and `best.json`.

The implementation is **GRAM-inspired** and **MDT-inspired**. It is not a full
reproduction of either method. The relevant research lineage includes:

- [Generative Recursive Reasoning (GRAM)](https://arxiv.org/abs/2605.19376)
- [Multi-view diffusion geometry using intertwined diffusion trajectories (MDTs)](https://arxiv.org/abs/2512.01484)

### Recursive Reasoning Experiment Ladder

RG-Tracer also exposes an additive ladder for controlled comparisons:

| Sampler | Purpose |
| --- | --- |
| **`trm`** | Deterministic Tiny Recursion Model baseline. This remains the default. |
| **`ptrm`** | PTRM-inspired bounded Gaussian test-time widening. |
| **`lattice_trm`** | Deterministic TRM with explicit task-local lattice projection. |
| **`lattice_ptrm`** | Widened trajectories constrained by explicit task-local lattice projection. |
| **`gram_mdt`** | Separate GRAM-inspired widening and MDT-inspired public view-routing scaffold. |

Only explicit task-local lattices may run in `gated` mode. Recovered or inferred
representation lattices remain shadow-only. Lattice projection cannot override
hard gates, semantic verification, repair, abstention, value decomposition,
overwatches, attribution, or Pareto selection.

Relevant papers:

- [Less is More: Recursive Reasoning with Tiny Networks (TRM)](https://arxiv.org/abs/2510.04871)
- [Probabilistic Tiny Recursive Model (PTRM)](https://arxiv.org/abs/2605.19943)

Detailed recursive-refinement documentation:

- [`reasoning-generalization-tracer/docs/recursive_refinement.md`](reasoning-generalization-tracer/docs/recursive_refinement.md)
- [`reasoning-generalization-tracer/docs/recursive_reasoning_ladder.md`](reasoning-generalization-tracer/docs/recursive_reasoning_ladder.md)
- [`reasoning-generalization-tracer/docs/lattice_deduction.md`](reasoning-generalization-tracer/docs/lattice_deduction.md)

### Additive LRD and Semantic-Constraint Experiments

`scripts/run_grokking_matrix.py` remains a lightweight null-backend attribution
matrix for smoke tests and attribution-shape checks. The separate optional
`scripts/run_lrd_grokking_matrix.py` path trains a small CPU toy transformer and
measures spectra from actual Q/K matrices. Install
`pip install -e "reasoning-generalization-tracer[grokking]"` to enable Torch.

The semantic-constraint scaffold compiles bounded synthetic requirements such
as "the answer must be even and greater than 2" into provenance-tracked finite
candidate constraints. Open-ended language remains ungated, generated code and
real external actions are out of scope, and recovered embedding lattices remain
shadow-only.

See:

- [`reasoning-generalization-tracer/docs/low_rank_decay_grokking.md`](reasoning-generalization-tracer/docs/low_rank_decay_grokking.md)
- [`reasoning-generalization-tracer/docs/semantic_constraint_synthesis.md`](reasoning-generalization-tracer/docs/semantic_constraint_synthesis.md)

## Circuit Traces, Concept Rewards, and Attribution Graphs

Concepts are defined through `ConceptSpec` entries with definitions, tests, and
expected substructures. Circuit traces are normalized before scoring:

- **Match**: expected tags appear on entailed features;
- **Selectivity**: attribution mass concentrates on relevant features rather than
  distractors;
- **Parsimony**: concept-specific paths stay compact;
- **Transfer**: concepts reappear across supporting tasks.

Attribution graphs log:

- sparsity (`HHI`);
- average path length;
- branching factor;
- repeatability across probes;
- concept alignment;
- delta metrics comparing phases such as `overfit` and `post_grok`.

The package supports the external
[`openai/circuit-tracer`](https://github.com/openai/circuit-tracer) dependency
when installed and a deterministic stub backend for CI and lightweight tests.

## Repository Layout

The main code lives under [`reasoning-generalization-tracer/`](reasoning-generalization-tracer/).
Important components include:

- `src/rg_tracer/scoring/` - eleven-axis rubric, profiles, and aggregators;
- `src/rg_tracer/runners/` - self-play and evaluation orchestration;
- `src/rg_tracer/concepts/` - concept specifications and circuit-reward logic;
- `src/rg_tracer/abstention/` - confidence calibration and abstention policy;
- `src/rg_tracer/schema_v3/` - 17-case V3 overlay and structured diagnostics;
- `src/rg_tracer/fallback/` - academic-to-Bayesian fallback for non-verifiable
  domains;
- `src/rg_tracer/recursive_refinement/` - bounded GRAM-inspired and MDT-inspired
  refinement scaffold;
- `src/rg_tracer/grokking/` - optional Torch-backed toy LRD grokking ablations;
- `src/rg_tracer/semantic_constraints/` - bounded rule-based semantic compiler
  for synthetic finite-domain tasks;
- `src/rg_tracer/trm_baseline/` - Tiny Recursion Model baseline;
- `src/rg_tracer/dapo/` - adapters for hybrid training with `gepa-dapo-grn`;
- `tests/` - unit tests for scoring, aggregation, abstention, concept rewards,
  schema overlays, self-play, and recursive baselines.

## Research Status and Limitations

RG-Tracer is an experimental research scaffold. Current limitations include:

- circuit tracing falls back to deterministic stubs when the external dependency
  is not installed;
- semantic verification uses lightweight heuristics rather than a full learned
  natural-language-inference model;
- Bayesian posteriors for non-verifiable domains are only as good as the stated
  priors, likelihood estimates, evidence quality, and humanities-rigor checks;
- the default TRM experiments are small toy tasks;
- `gram_mdt` is a heuristic CPU-friendly scaffold, not a learned GRAM-style
  trajectory prior;
- MDT-inspired routing is not a mathematical diffusion-geometry implementation;
- PTRM-inspired and lattice samplers are controlled toy experiments rather than
  faithful reproductions of every source architecture;
- recursive-refinement interfaces are designed for future learned backends, but
  the current code does not claim LLM-scale recurrent reasoning.
- LRD experiments are LRD-inspired toy runs over small matrices, not faithful
  reproduction claims unless future work matches the reference setup closely.
- semantic compilation is bounded and synthetic; arbitrary natural language
  cannot gate outputs or control external systems.

## Roadmap Boundaries

**CURRENT IMPLEMENTATION:** deterministic TRM -> PTRM-inspired bounded widening
-> LDT-inspired explicit finite candidate lattices -> lattice-constrained PTRM
-> GRAM/MDT-inspired public multi-view refinement.

**THIS PR:** real optional LRD grokking ablations and bounded semantic
constraint compilation into explicit toy lattices.

**FUTURE SHADOW MODE:** recovered embedding-space concept lattices,
representation-lattice validation, and mechanistic-interpretability comparison.

**FUTURE RESEARCH:** learned GRAM-style trajectory priors, adaptive MDT route
learning, category-theoretic mappings across typed lattices, and LLM-scale
recurrent reasoning.

## Selected References

- [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)
- [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)
- [Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences](https://arxiv.org/abs/2511.02109)
- [Atom-Searcher: Enhancing Agentic Deep Research via Fine-Grained Atomic Thought Reward](https://arxiv.org/abs/2508.12800)
- [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
- [Probabilistic Tiny Recursive Model](https://arxiv.org/abs/2605.19943)
- [Lattice Deduction Transformers](https://arxiv.org/abs/2605.08605)
- [The Lattice Representation Hypothesis of Large Language Models](https://arxiv.org/abs/2603.01227)
- [Low-Rank Decay for Grokking in Scale-Invariant Transformers](https://arxiv.org/abs/2606.04405)
- [Semantic Constraint Synthesis for Adaptive Trajectory Optimization via Large Language Models](https://arxiv.org/abs/2606.04123)
- [Generative Recursive Reasoning](https://arxiv.org/abs/2605.19376)
- [Multi-view diffusion geometry using intertwined diffusion trajectories](https://arxiv.org/abs/2512.01484)
- [GeoCert: Certified Geometric AI for Reliable Forecasting](https://arxiv.org/abs/2604.23474)
