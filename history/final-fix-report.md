# Scoped final review fixes

Scope: both findings in `history/final-review.md`, plus the parent's follow-up request to
close the public `CanonicalIdentity` constructor bypass. No commits, staging, or subagents.

## Finding 1: contradictory graph edges bypassed claim safeguards

`OntologyGraph.add_relation()` now validates the entire insertion, computes affected claim
updates, then appends the edge and updates claims atomically. All evidence-field bookkeeping
lives in this path; `record_evidence()` delegates to it. Direct and adapter-created EVIDENCE
SUPPORTS/CONTRADICTS CLAIM edges populate the corresponding evidence IDs without duplicates.

A contradiction marks the claim CONTRADICTED and appends the previous status only when the
status changes. Previous support, verification IDs, edge records, metadata, and provenance
remain intact. A Claim CONTRADICTS Claim edge disputes both endpoints while preserving the
original directed assertion; neither Claim ID is counted as task-level Evidence.

Promotion and export independently inspect contradiction edges as well as explicit claim
evidence lists. They cannot certify a disputed claim even if a reconstructed/corrupted claim
projection has stale status or evidence fields. Direct SUPPORTS edges now support promotion
when an independent claim-bound task verification is present.

Regression evidence: the new direct-edge/adapter cases first reproduced **7 failures**.
After implementation, the suite covers direct support promotion; contradiction before/after
promotion; claim-to-claim contradiction in either direction; stale verified export; malformed
edge atomicity; and reasoning-adapter contradictory evidence. The invalid-edge atomicity test
also passed before the fix and continues passing.

## Finding 2: obsolete ambiguity-case migration note

`docs/17_case_framework.md` now states that Cases 14–17 are canonical V5 ambiguity-handling
cases, selected with explicit ambiguity mode and stakes metadata. They do not replace cases
9–13; existing case IDs remain stable.

Verification: `rg` found the corrected text at line 240 and no `aliases only` occurrence in
that document. A shared-contract import assertion verified that IDs 14–17 are canonical.

## Follow-up: direct canonical identity construction

`CanonicalIdentity.__post_init__()` copies observations into a tuple and invokes the shared
identity validator. `canonicalize_claims()` returns this validated public constructor instead
of maintaining a separate validation implementation. Both paths enforce non-empty identity,
at least two distinct Claim observations, the declared shared semantic key, equal scoped
constraints, and a passed equivalence verification bound to exactly those observations.

Regression evidence: **9 tests failed before the fix**, covering eight invalid constructions
and mutation of the original observation list. They now pass; original claim provenance stays
available after the caller mutates its input list.

## Documentation and scope

Updated `docs/ONTOLOGY.md` and the reasoning adapter's docstring to accurately describe direct
edge bookkeeping, disputed claims, retained verification history, and validated constructor
behavior. No other capabilities or scoring policies changed.

## Final verification

All commands ran from `reasoning-generalization-tracer`, using
`C:/Users/evanh/Documents/Codex/v5env-gepa/Scripts/python.exe` and `PYTHONPATH=src` for tests.

- Ontology-specific suite: **55 passed in 3.68 seconds**.
- Final combined ontology and existing reasoning/lattice/attribution/semantic compatibility
  suite: **86 passed in 4.53 seconds**.
- Black `--check --line-length 100`: all nine ontology/test Python files unchanged, exit 0.
- Ruff `--select F,E501 --line-length 100`: all checks passed, exit 0.
- Shared canonical-ID assertion and corrected migration-note search: passed.

Combined test command:

```text
python -m pytest tests/test_ontology.py tests/test_ontology_adapters.py
  tests/test_reasoning_graphs.py tests/test_concept_lattice_registry.py
  tests/test_lattice_core.py tests/test_lattice_adapters.py
  tests/test_attribution_metrics.py tests/test_attribution_integration.py tests/test_semantics.py -q
```

Ready for the parent's scoped independent re-review. Full integrated tests and final wheel
verification remain under the parent's coordination.
