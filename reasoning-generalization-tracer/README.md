# Reasoning Generalization Tracer

Reasoning Generalization Tracer (RG-Tracer) is a research toolkit for studying
reasoning, abstraction, and concept reuse through self-play. The project
combines a deterministic multi-axis rubric, circuit-level concept rewards,
confidence-aware abstention, a Tiny Recursion Model (TRM) baseline, and a
semantic verifier with targeted repair. All components are lightweight,
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
and penalising contradictory reuse.

## Abstention at 0.75 Confidence

The abstention policy enforces a calibrated threshold of `0.75`. Any candidate
with lower confidence, failing semantic checks (score < 2), or blocked by hard
gates emits the exact string “I don't know.” No token reward is granted; a
separate “honesty” metric can be layered on in downstream analysis.

## Semantic Verifier and Repair

`verify_chain` flags contradictions, unsupported claims, unit mismatches,
variable drift, and definition drift. It returns per-step tags plus a 0–4 score.
When the score falls below 2 (or confidence < 0.75) the pipeline applies a
single `repair_once` pass to add missing units, relabel drifting variables, or
justify steps. If the repaired chain still fails, the system abstains.

Metrics are logged to `runs/<timestamp>/semantics.jsonl` alongside summary
artifacts, enabling downstream auditing of contradiction rates, entailed steps,
and repair success.

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
[`circuit-tracer`](https://github.com/safety-research/circuit-tracer) project
pinned in `pyproject.toml`.

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

## Expected Outputs

Each self-play run emits:

* `scores.jsonl` – per-candidate rubric scores and rewards.
* `semantics.jsonl` – semantic report with contradiction rates, unit checks, repairs.
* `summary.md` – table covering composite, concept reward, abstentions.
* `best.json` – best-performing candidate under the chosen profile.

## Configuration

* **Profiles:** tweak weights in `profiles.yaml` or supply a custom file.
* **Concept Rewards:** override weights via the `weights` parameter in
  `compute_concept_reward`.
* **Abstention:** calibrate model confidences using `abstention/calibrate.py`.
* **Semantic Repair:** customise behaviour by editing `semantics/repair.py` and
  `semantics/verifier.py` heuristics.

## Limitations

* Circuit tracing falls back to a stub if the external dependency is missing.
* Semantic checks rely on lightweight heuristics rather than full NLI models.
* The TRM sampler is tuned for small parity/carry tasks and will not scale to
  complex domains without additional training.

## Roadmap

1. Integrate large-language-model samplers with calibration hooks.
2. Expand semantic verifier with learned contradiction detectors.
3. Add richer datasets covering physics and program synthesis tasks.
4. Expose a web dashboard for inspecting run artifacts.
