# Recursive Reasoning Experiment Ladder

RG-Tracer uses a ladder of bounded CPU-friendly samplers instead of a monolithic architecture. Each
rung isolates one mechanism so runs can compare deterministic recursion, stochastic widening,
explicit deduction projection, their combination, and the existing adaptive multi-view scaffold.

## Rungs

- `trm`: deterministic toy Tiny Recursion Model baseline. This remains the default.
- `ptrm`: PTRM-inspired test-time widening with seeded bounded Gaussian perturbations. It does not
  retrain a model and is not a faithful PTRM reproduction.
- `lattice_trm`: deterministic TRM candidate generation with explicit task-local LDT-inspired
  projection over finite candidate sets.
- `lattice_ptrm`: PTRM-inspired widened trajectories constrained by explicit task-local lattice
  projection. Contradicted branches are pruned and equivalent branches can be merged.
- `gram_mdt`: the existing GRAM-inspired recursive-refinement scaffold with MDT-inspired public
  view routing. It remains a separate comparison branch.

## Safety And Scope

Only explicit task-local lattices may run in `gated` mode. Recovered or inferred representation
lattices are shadow-only in this increment. Lattice projection cannot override hard gates, semantic
verification failures, repair, abstention, GRN hooks, value decomposition, overwatch, attribution, or
Pareto selection.

Contradiction and unresolved states recommend abstention in gated mode because an empty candidate
set means the public constraints are inconsistent, while a multi-candidate set means bounded
deduction has not selected a unique answer.

## Reproduction

```bash
rg-tracer self-play \
  --profile proof_math \
  --problem datasets/toy_math/addition_small.jsonl \
  --sampler ptrm \
  --trajectory-count 8 \
  --noise-std 0.10 \
  --seed 7
```

```bash
rg-tracer self-play \
  --profile proof_math \
  --problem datasets/constraint_lattice/finite_domain.jsonl \
  --sampler lattice_ptrm \
  --lattice-mode gated \
  --trajectory-count 8 \
  --noise-std 0.10 \
  --seed 7
```

```bash
python scripts/run_recursive_ladder.py \
  --dataset datasets/constraint_lattice/finite_domain.jsonl \
  --seeds 0 1 2
```

The matrix writes `results.jsonl`, `summary.csv`, and `summary.md` under
`runs/recursive_ladder/<timestamp>/`. Relevant sampler runs also write
`lattice_diagnostics.jsonl`, `perturbations.jsonl`, and `ladder_metrics.json`.

## Non-Implementations

This increment does not implement a learned GRAM-style trajectory prior, mathematical MDT diffusion
geometry, category-theoretic mappings, attribution-verified concept transitions, recovered
embedding lattices for gating, or LLM-scale recurrent reasoning. It also does not add negative
hidden-thought penalties or collapse decomposed rewards into one opaque lattice reward.
## Roadmap Boundary

Current implementation: deterministic TRM -> PTRM-inspired bounded widening ->
LDT-inspired explicit finite candidate lattices -> lattice-constrained PTRM ->
GRAM/MDT-inspired public multi-view refinement. This PR adds optional real LRD
toy grokking ablations and bounded semantic constraint compilation into
explicit toy lattices. Future shadow-mode work may study recovered
embedding-space concept lattices, but those recovered lattices cannot gate
outputs.
