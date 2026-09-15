# V5 behavioral authority and the reasoning ontology

Status: accepted for this migration.

## Context

DSPy's local case-name table drifted from Mindfulness V5. Reasoning, lattice and
attribution graphs describe different evidence channels without shared typed identity.
Experiments need reproducible contracts and interoperable metadata without requiring
a network connection or replacing specialized structures.

## Decision

Mirror Mindfulness's YAML byte-for-byte as versioned package resources, pin its source
commit and SHA-256 hashes, and derive Python identities through one loader. Explicit
maintainer synchronization remains separate from runtime and unit tests.

Retain `CaseV3Result` and all V3 overlays. V5 controls behavioral identity, while DSPy
continues to implement its historical numeric reward policy. Correct two demonstrated
ambiguity routing mismatches: low-stakes assumptive proceeding and repeated questions.

Introduce `rg-ontology-v1` as a registry of typed entities and relations with adapters.
Separate semantic, execution/provenance, evaluation/failure and mechanistic graph views.
Preserve observations and provenance when recording candidate or verified equivalence.
Do not promote mechanistic observations to semantic truth or reward ontology vocabulary.

## Alternatives and consequences

Runtime GitHub reads would make old runs dependent on upstream changes and connectivity.
A second hardcoded Python table would recreate drift. Replacing existing graphs would
expand compatibility risk and discard specialized semantics. A universal graph DAG would
reject meaningful semantic cycles. The chosen design instead adds small adapters and
requires callers to supply explicit verification and provenance.

V5, ontology and overlay versions are independent. Older names load through an explicit
adapter, while new records use canonical names. Pinned provenance mismatches fail rather
than silently relabeling an experiment. Beads bd-6.4 tracks reward-policy follow-up.

## Verification

`tests/test_epistemic_cases.py`, `tests/test_v5_integration.py` and ontology tests check
the identities, routing, provenance, adapters and graph boundaries. Existing tests remain
the compatibility check for specialized structures and historical reward values.
