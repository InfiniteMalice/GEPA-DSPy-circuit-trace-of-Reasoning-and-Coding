# Open-World Tool Generalization

## Summary

Open-world utilities generate deterministic perturbations for queries, tools, observations, and
domains.

## Reference

Tool-use shifts follow OpenAgent-style tests (Lv et al., 2026; arXiv:2607.01084). Role-filler and
irrelevant-detail perturbations follow pattern-matching robustness tests (Studdiford and Lupyan,
2026; arXiv:2606.13607).

## Why It Belongs

RG-Tracer can test whether reasoning workflows overfit fixed prompts, schemas, observations, or
surface domains.

## Implemented

- Query perturbations for irrelevant details, paraphrases, and ambiguity.
- Tool/schema perturbations for renames, field ordering, optional fields, and distractors.
- Observation perturbations for stale, redundant, reformatted, and anomalous observations.
- Domain perturbations for surface swaps and role-filler substitutions.

## Future Work

These records can later feed eval suites and abstention calibration.

## Compatibility Notes

Open-world perturbations test robustness and abstention behavior under shift. They do not call any
external tool or model.
