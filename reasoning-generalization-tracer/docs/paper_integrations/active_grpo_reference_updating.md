# Active-GRPO Reference Updating

## Summary

Active-GRPO support is implemented as an optional reference-management strategy.

## Reference

Active reference updating follows Active-GRPO (Liu et al., 2026; arXiv:2607.00531).

## Why It Belongs

RG-Tracer can compare verified candidates against references without declaring any optimizer to be
the project default.

## Implemented

- `ReferenceRecord`, `CandidateRecord`, and `ActiveGRPODecision` dataclasses.
- `decide_active_grpo` for imitate, reinforce, reject, and no-reference decisions.
- `maybe_update_reference` with lineage preservation.
- Append-only JSONL reference history.

## Future Work

Adapters can connect this strategy to GEPA, GRPO, DAPO-compatible optimization, PPO+GRN, or future
GEPA-DAPO-GRN loops.

## Compatibility Notes

Active-GRPO is not a replacement for GEPA, DAPO, GRN, PPO, or ordinary GRPO. It is disabled unless
explicitly imported and invoked.
