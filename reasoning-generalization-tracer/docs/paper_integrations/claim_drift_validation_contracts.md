# Claim Drift And Validation Contracts

## Summary

Validation contracts and claim-drift reports track whether artifacts stay inside their evidence
boundaries.

## Reference

Claim drift and validation contracts follow XCIENTIST-style research harness ideas (Wang et al.,
2026; arXiv:2606.18874).

## Why It Belongs

RG-Tracer can pass narrow checks while an answer or patch overclaims. Explicit contracts make those
boundaries inspectable.

## Implemented

- `ValidationContract` with evidence, test, shortcut, and acceptance fields.
- Deterministic contract validation.
- `ClaimDriftReport` generation from provided diagnostics.
- `RepairTrace` JSON serialization.

## Future Work

Future harnesses can map paper graphs, repair traces, and experiment results into contract
diagnostics.

## Compatibility Notes

Claim-drift checks prevent final artifacts from overclaiming beyond evidence. Core functions do not
make LLM calls.
