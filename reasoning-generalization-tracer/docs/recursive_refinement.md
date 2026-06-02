# Experimental Recursive Refinement

This document describes the additive `sampler="gram_mdt"` scaffold in
RG-Tracer. The default self-play sampler remains `sampler="trm"`.

## What GRAM-Inspired Means

The sampler is inspired by the research idea of recursively refining a latent or
structured reasoning state and allocating extra compute only when the process
looks uncertain, divergent, novel, or higher stakes. In this repository that is
implemented as a small deterministic/seeded controller over public toy-task
state. It is not a faithful GRAM reproduction and does not implement a learned
variational prior.

## What MDT-Inspired Means

The sampler routes trajectories through sequences of view-specific refinement
operators. The initial views are:

- `arithmetic`
- `parity`
- `constraint_check`
- `counterexample_check`
- `verification`
- `compression`

This is MDT-inspired multi-view routing, not a mathematical diffusion-geometry
implementation. Each view records public metadata: selected view, operation
name, order, revisit status, prediction changes, uncertainty reduction,
confidence increases, and constraint/verification status.

## How Adaptive Depth Works

The controller begins with a cheap shallow pass over the toy problem, then
recursively applies one public view at a time. Adaptive halting stops a
trajectory when confidence is high and uncertainty is low, independent branches
converge, no relevant view remains, a branch is pruned, depth is exhausted, or
the update budget is exhausted.

## How Progressive Widening Works

A seeded RNG creates bounded alternative trajectories when uncertainty or
disagreement crosses configured thresholds. Widening is capped by `max_width`,
recursion is capped by `max_depth`, and the whole run is capped by
`max_total_updates`. Shared prefixes are represented in trajectory histories but
are counted once in run-level update accounting.

## Process Scores

Process scores are heuristic and decomposed. They include final confidence,
uncertainty reduction, verification, convergence, efficiency, view diversity,
redundancy, contradiction counts, and budget diagnostics. They are not a learned
latent process reward model. The score can affect ranking only through
`process_reward_weight`, which defaults to `0.0`.

Hard gates remain authoritative. Process bonuses are not applied to candidates
that fail gates or abstain, and no negative reward is applied directly to hidden
thought traces. Existing positive-only thought reward, GRN flags, value
decomposition, overwatch, abstention, semantic repair, attribution, and
circuit-trace compatibility remain intact.

## Artifacts

`sampler="gram_mdt"` writes the standard self-play artifacts and adds:

- `trajectories.jsonl`: public trajectory states and final summaries.
- `view_routes.jsonl`: per-trajectory view route and operation records.
- `budget_metrics.json`: total updates, max depth, max width, branch count,
  prune count, convergence, and budget exhaustion.
- `summary.md`: includes a concise Recursive Refinement Metrics section.

Candidate records in `scores.jsonl` and `best.json` also include
`trajectory_id`, `trajectory_metadata`, `process_score`,
`process_score_components`, `view_route`, `total_updates`, `max_depth`,
`max_width`, `converged`, and `budget_exhausted`.

## Reproducibility

Use `--seed` for deterministic routes and perturbations:

```bash
rg-tracer self-play \
  --profile proof_math \
  --problem datasets/toy_math/addition_small.jsonl \
  --concept parity \
  --sampler gram_mdt \
  --k 4 \
  --max-depth 6 \
  --max-width 4 \
  --max-total-updates 48 \
  --seed 7
```

Disable components for ablations:

```bash
rg-tracer self-play \
  --profile proof_math \
  --problem datasets/toy_math/addition_small.jsonl \
  --sampler gram_mdt \
  --disable-adaptive-halting \
  --disable-progressive-widening \
  --disable-view-routing
```

## Limitations

1. The current sampler is heuristic and CPU-friendly.
2. It is not a learned variational prior.
3. It is not a full latent process reward model.
4. It is not a full MDT diffusion-geometry implementation.
5. It does not yet scale to LLM inference.
6. The interfaces are designed so learned backends can be added later.
