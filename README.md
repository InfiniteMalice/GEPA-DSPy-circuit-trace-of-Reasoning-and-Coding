# GEPA DSPy: Circuit Trace of Reasoning and Coding

This repository hosts the **Reasoning Generalisation Tracer (RG-Tracer)**
scaffold, a research playground for evaluating and improving reasoning
systems. It combines multi-axis reasoning fitness scoring, self-play
optimization, circuit-level concept rewards, calibrated abstention, and a Tiny
Recursion Model baseline. The code lives under
[`reasoning-generalization-tracer/`](reasoning-generalization-tracer/) and is
packaged as a Python project exposing the `rg-tracer` command line interface.

## Repository Layout

- `reasoning-generalization-tracer/` – main Python package and tests.
- `README.md` – this document.

Inside the package you will find the following important components:

- `src/rg_tracer/scoring/` – eleven-axis reasoning rubric, profiles, and
  aggregators.
- `src/rg_tracer/runners/` – self-play and evaluation orchestration.
- `src/rg_tracer/concepts/` – circuit tracer adapters and concept reward logic.
- `src/rg_tracer/abstention/` – calibration helpers and the abstention policy
  enforcing the 0.75 confidence threshold.
- `src/rg_tracer/trm_baseline/` – Tiny Recursion Model implementation,
  training, and evaluation utilities.
- `tests/` – unit tests covering the scoring rubric, aggregation, self-play,
  concept rewards, abstention, and TRM baseline.

For detailed documentation, installation steps, and usage examples please refer
to [`reasoning-generalization-tracer/README.md`](reasoning-generalization-tracer/README.md).

## Getting Started

Clone the repository and install the package in editable mode:

```bash
pip install -e reasoning-generalization-tracer
```

After installation the following commands are available:

- `rg-tracer self-play` – run self-play with scoring, abstention, and concept
  rewards.
- `rg-tracer eval` – batch evaluation across datasets with CSV reports.
- `rg-tracer trace` – export circuit traces and concept reward summaries.

Refer to the package README for dataset locations, configuration examples, and
roadmap details.
