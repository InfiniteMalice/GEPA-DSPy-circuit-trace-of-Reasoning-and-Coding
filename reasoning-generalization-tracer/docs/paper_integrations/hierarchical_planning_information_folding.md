# Hierarchical Planning And Information Folding

## Summary

Hierarchical state utilities fold completed subgoals into compact summaries while preserving open
constraints and failed attempts.

## Reference

Planning and folding follow HIPIF (Diao et al., 2026; arXiv:2606.10507) and hierarchical memory
navigation follows HORMA (Hsu et al., 2026; arXiv:2606.11680).

## Why It Belongs

Long RG-Tracer runs need compact state without losing active constraints or provenance.

## Implemented

- `Subgoal` records.
- `FoldedState` JSON serialization.
- `fold_subgoals` for completed summaries, active subgoals, unresolved constraints, failed
  attempts, and evidence refs.

## Future Work

Future memory adapters can add navigation over raw trajectory provenance.

## Compatibility Notes

Hierarchical folding must not drop unresolved constraints.
