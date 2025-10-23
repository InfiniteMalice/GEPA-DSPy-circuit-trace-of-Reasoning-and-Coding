# Reasoning Generalization Tracer

Reasoning Generalization Tracer (RG-Tracer) is a research toolkit for studying
reasoning, abstraction, and concept reuse through self-play. The project
combines a deterministic multi-axis rubric, circuit-level concept rewards,
confidence-aware abstention, a Tiny Recursion Model (TRM) baseline, a humanities
rigor module, and a semantic verifier with targeted repair. When problems lack
verifiable reward signals, RG-Tracer falls back to an academic three-step
pipeline culminating in a Bayesian position. All components are lightweight,
CPU-friendly, and designed for reproducible experimentation.

## Rubric Axes

Scores are deterministic integers from 0 to 4 across eleven axes:

1. **Logical Validity** – contradictions and formal proof guarantees.
2. **Conceptual Clarity** – undefined symbols and naming consistency.
3. **Completeness** – edge/boundary handling and coverage ratios.
4. **Rigor** – proportion of steps justified by checks.
5. **Efficiency** – step count relative to a baseline.
6. **Heuristic Creativity** – diversity and novelty of solution paths.
7. **Numerical Accuracy** – tolerance-aware error measurement.
8. **Cognitive Efficiency** – token/time/memory budgets.
9. **Explanatory Power** – causal links and illustrative examples.
10. **Self Consistency** – self-critique and repair success.
11. **Abstraction & Generalization** – transfer, compression, lifts, lemmas.

Profiles in `src/rg_tracer/scoring/profiles.yaml` weight axes via a geometric
mean with hard gates on logical validity, rigor, and numerical accuracy.

## Concept Rewards

Concepts are defined via `ConceptSpec` entries that include definitions, tests,
and expected substructures. Circuit traces (from `circuit-tracer` or the stub
adapter) are normalised before computing a reward:

* **Match** – expected tags appear on entailed features.
* **Selectivity** – importance mass on target features vs distractors.
* **Parsimony** – shorter concept-specific paths.
* **Transfer** – reuse across supporting tasks.

Semantic reports inform the reward by filtering for entailed/supported features
and penalising contradictory reuse. Circuit traces are produced via
`circuit-tracer` (pinned) or the bundled stub; the adapter raises a clear error
if the dependency is absent.

## Abstention at 0.75 Confidence

The abstention policy enforces a calibrated threshold of `0.75`. Any candidate
with lower confidence, failing semantic checks (score < 2), or blocked by hard
gates emits the exact string “I don't know.” No token reward is granted; a
separate “honesty” metric can be layered on in downstream analysis.

## Semantic Verifier and Repair

`verify_chain` flags contradictions, unsupported claims, unit mismatches,
variable drift, definition drift, rhetorical excess, misquotes, uncited claims,
over-claimed causality, and is-ought slips. It returns per-step tags plus a
0–4 score. When the score falls below 2 (or confidence < 0.75) the pipeline
applies a single `repair_once` pass to add missing units, relabel drifting
variables, insert citations, hedge causal claims, or note normative statements.
If the repaired chain still fails, the system abstains.

Metrics are logged to `runs/<timestamp>/semantics.jsonl` alongside summary
artifacts, enabling downstream auditing of contradiction rates, entailed steps,
and repair success.

## Humanities Rigor Scoring

Humanities problems are scored along twelve axes covering source handling,
interpretive fidelity, historiography, causal discipline, triangulation,
normative/positive separation, uncertainty calibration, intellectual charity,
rhetorical hygiene, reproducibility, synthesis, and epistemic neutrality. Hard
gates enforce minimum thresholds on source handling, interpretive fidelity, and
normative clarity before any reward is granted. Signals include citation
coverage, quote integrity, counterevidence ratios, hedging, fallacy flags, and
neutrality balance.

## Tiny Recursion Model Baseline

The TRM is a compact recursive network that emits intermediate “thought states.”
`trm_train.py` offers CPU-friendly training loops for parity and carry tasks,
while `trm_eval.py` reports accuracy, per-axis means, semantic scores, and
stored traces for inspection. The TRM also drives the default self-play sampler
for toy tasks.

## Installation

```bash
pip install -e .
```

Python 3.10+ is required. Circuit tracing depends on the
[`circuit-tracer`](https://github.com/openai/circuit-tracer) project pinned in
`pyproject.toml`.

## Quickstart

Run self-play on the toy addition dataset using the TRM sampler:

```bash
rg-tracer self-play --profile proof_math --k 8 \
  --problem datasets/toy_math/addition_small.jsonl \
  --concept parity --sampler trm
```

Inspect the generated run directory for `scores.jsonl`, `summary.md`,
`semantics.jsonl`, and `best.json`. Generate a concept trace for the best
candidate:

```bash
rg-tracer trace --run runs/<timestamp>/best.json --out runs/<timestamp>/circuits/ \
  --concept parity
```

Batch-evaluate transfer datasets and write a CSV summary:

```bash
rg-tracer eval --dataset "datasets/transfer_tests/*.jsonl" --profile proof_math \
  --output transfer_eval.csv
```

Run humanities scoring or the fallback pipeline:

```bash
rg-tracer humanities --dataset datasets/humanities/sample_claims.jsonl \
  --profile humanities

rg-tracer fallback --problem datasets/humanities/sample_claims.jsonl
```

## Expected Outputs

Each self-play run emits:

* `scores.jsonl` – per-candidate rubric scores and rewards.
* `semantics.jsonl` – semantic report with contradiction rates, humanities metrics, repairs.
* `summary.md` – table covering composite, concept reward, abstentions.
* `best.json` – best-performing candidate under the chosen profile.

## Configuration

* **Profiles:** tweak weights in `profiles.yaml` or supply a custom file.
* **Concept Rewards:** override weights via the `weights` parameter in
  `compute_concept_reward`.
* **Abstention:** calibrate model confidences using `abstention/calibrate.py`.
* **Semantic Repair:** customise behaviour by editing `semantics/repair.py` and
  `semantics/verifier.py` heuristics.
* **Humanities Profiles:** adjust humanities weights in
  `humanities/profiles.yaml`.
* **Fallback:** extend the Bayesian priors/likelihoods in `fallback/bayes.py`.

## Limitations

* Circuit tracing uses a local stub when the external dependency is missing.
* Semantic checks rely on lightweight heuristics rather than full NLI models.
* The humanities module encourages citations but does not fetch external
  sources automatically.
* The TRM sampler is tuned for small parity/carry tasks and will not scale to
  complex domains without additional training.

## Roadmap

1. Integrate large-language-model samplers with calibration hooks.
2. Expand semantic verifier with learned contradiction detectors.
3. Add richer datasets covering physics and program synthesis tasks.
4. Expose a web dashboard for inspecting run artifacts.
