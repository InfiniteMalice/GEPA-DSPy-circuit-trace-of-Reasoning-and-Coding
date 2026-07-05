# Agentic Transaction Processing

## Summary

Generated artifacts can be represented as proposals that require deterministic admission.

## Reference

Admission control follows Mnemosyne/ATP (Chang et al., 2026; arXiv:2607.00269).

## Why It Belongs

RG-Tracer often compares answers, repairs, workflow steps, and reference updates. Treating these as
proposals keeps accepted state separate from unverified generation.

## Implemented

- Generic `Proposal` and `AdmissionDecision` dataclasses.
- Constraint registry with required-field, verifier, score, contradiction, claim-drift, and
  provenance checks.
- Append-only JSONL transition log.
- Effective-state projection over accepted transitions.

## Future Work

Future adapters can attach admission decisions to repair controllers or optimizer state updates.

## Compatibility Notes

Transaction/admission control is a runtime safety boundary, not a measure of model virtue.
