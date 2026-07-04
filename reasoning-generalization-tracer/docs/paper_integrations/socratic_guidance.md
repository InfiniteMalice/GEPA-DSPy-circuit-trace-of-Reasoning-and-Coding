# Socratic Guidance

## Summary

Socratic guidance support records assisted attempts and measures dependence on guidance.

## Reference

Guidance metadata follows SocraticPO (Liu et al., 2026; arXiv:2606.09887) and Socratic Agents for
physical science (Zeng et al., 2026; arXiv:2606.26722).

## Why It Belongs

RG-Tracer can distinguish independent success from teacher-assisted recovery.

## Implemented

- `GuidanceRecord` and `AssistedAttemptRecord` dataclasses.
- Assisted improvement and reward-decay helpers.
- Teacher-dependency metrics.

## Future Work

No teacher model is implemented. Future adapters can record critic guidance from external systems.

## Compatibility Notes

Socratic guidance must track dependency so the model does not learn to rely on hints.
