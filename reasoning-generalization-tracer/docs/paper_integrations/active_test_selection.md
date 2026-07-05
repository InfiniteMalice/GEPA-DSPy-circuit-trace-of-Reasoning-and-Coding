# Active Test Selection

## Summary

Active selection utilities prioritize diagnostic tasks where candidate predictions disagree.

## Reference

Active diagnostic selection follows ATLAS (Elteto et al., 2026; arXiv:2606.12386).

## Why It Belongs

RG-Tracer often compares candidate hypotheses, repairs, or policies. Disagreement can identify
examples that separate them.

## Implemented

- Disagreement scoring across candidate predictions.
- Deterministic top-k diagnostic task selection.
- Selection metadata explaining why a task was chosen.

## Future Work

This does not train models or run an evolutionary optimizer. Future adapters can connect selected
tasks to experiment harnesses.

## Compatibility Notes

The module is optional, deterministic, and handles empty candidate sets safely.
