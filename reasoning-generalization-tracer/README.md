# Reasoning Generalisation Tracer

Reasoning Generalisation Tracer (RG-Tracer) is a lightweight research toolkit
for evaluating and improving reasoning systems. It couples a detailed
multi-axis reasoning rubric with self-play orchestration, circuit-level concept
rewards, and a confidence-calibrated abstention policy. The repository also
ships a Tiny Recursion Model (TRM) baseline that can be trained on CPU-sized
mathematical reasoning tasks.

## Features

* **Reasoning Fitness Rubric** – Deterministic 0–4 scoring across eleven axes
  including abstraction/generalisation.
* **Self-Play Runners** – Generate multiple candidates, score them, perform
  Pareto selection, and log artefacts under `runs/<timestamp>/`.
* **Circuit Concept Rewards** – Integrate with `circuit-tracer` to detect
  features supporting target concepts and compute rewards for correct reuse.
* **Confidence-aware Abstention** – Enforce a `0.75` threshold and output
  “I don't know.” when confidence is low.
* **Tiny Recursion Model Baseline** – CPU-friendly recursive model with toy
  training/evaluation pipelines for parity, carry, and related tasks.

## Installation

```bash
pip install -e .
```

Python 3.10+ is required. Circuit analysis relies on the
[`circuit-tracer`](https://github.com/openai/circuit-tracer) project pinned to a
specific commit. Optional language-model samplers can be enabled with
`pip install -e .[llm]`.

## Quickstart

Run a self-play experiment on the toy dataset and inspect the generated run
artefacts:

```bash
rg-tracer self-play --profile proof_math --k 4 \
  --problem src/rg_tracer/datasets/toy_math/addition_small.jsonl \
  --concept parity --sampler trm
```

Outputs are written to `runs/<timestamp>/` and include `scores.jsonl`,
`summary.md`, and `best.json`. Use the trace command to produce concept traces:

```bash
rg-tracer trace --run runs/<ts>/best.json --out runs/<ts>/circuits/ --concept parity
```

Batch dataset evaluation and export CSV summaries:

```bash
rg-tracer eval --dataset "src/rg_tracer/datasets/transfer_tests/*.jsonl" \
  --profile proof_math --output transfer_eval.csv
```

## Profiles & Scoring

Profiles live in `src/rg_tracer/scoring/profiles.yaml` and specify weighted
geometric means with hard gates (`logical_validity`, `rigor`,
`numerical_accuracy`). The default `proof_math` profile emphasises formal
correctness, while `scientific`, `heuristic`, and `efficiency` profiles
rebalance weights towards explanatory, creative, or efficient behaviours.

The scoring axes implement the rubric detailed in `axes.py`. Each axis consumes
interpretable metrics (e.g., contradiction counts, transfer accuracy) and
returns an integer score between 0 and 4.

## Concept Rewards

Concept specifications capture definitions, tests, and expected substructures.
During self-play, traces produced by `circuit-tracer` are normalised into a
consistent schema before computing:

* **Match** – expected tags found in the trace.
* **Selectivity** – importance of target features relative to distractors.
* **Parsimony** – path length efficiency when using the concept.
* **Transfer** – reuse of concept features across related tasks.

Weights default to `0.4/0.3/0.2/0.1` but can be overridden per-call.

## Confidence Calibration

`rg_tracer.abstention` exposes temperature scaling and isotonic regression
helpers. Calibrate confidences on a validation split, then call
`apply_abstention` to enforce the 0.75 threshold throughout pipelines. Honesty
rewards can be layered on downstream components when abstentions avoid errors.

## Tiny Recursion Model (TRM)

The TRM baseline is implemented in `trm_baseline/` and provides training and
evaluation utilities. A quick CPU-run that learns parity can be invoked via:

```bash
python -m rg_tracer.trm_baseline.trm_train
```

Evaluation dumps axis-aligned scores plus optional reasoning traces that
capture intermediate recursive activations.

## Roadmap

* Plug in stronger samplers (LLMs, heuristic search, symbolic provers).
* Expand circuit rewards with richer concept taxonomies and task metrics.
* Integrate abstention-aware reinforcement learning.
* Grow dataset coverage with harder algebra/proof benchmarks.

## Limitations

RG-Tracer is a research scaffold; the default samplers and datasets are toy
examples. Circuit tracing relies on the availability of the external
`circuit-tracer` dependency and currently returns stub traces when unavailable.
Concept rewards are heuristic and may require tuning for complex models.
