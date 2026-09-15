# Independent final whole-change review

Reviewed the implementation diff in `work/implementation-review.diff`, actual source,
the two binding user specifications (V5 sections 0–28 and ontology sections 29–61),
repository instructions, the implementation plan/report, and the repository quality gate.
The review also inspected the parent's subsequent self-play metadata preflight changes.
This report records the reviewed snapshot; subsequent fixes require scoped re-review.

## Verdicts

**Spec compliance: BLOCK pending two corrections.** The pinned behavioral contract,
compatibility migration, four ontology views, adapters, provenance, packaging, and separate
version axes are implemented. The ontology's conflicting-evidence gate can be bypassed
through its public relation API, and one required documentation audit retains a competing
description of canonical cases 14–17.

**Code quality: BLOCK pending finding 1.** The implementation is generally cohesive and
readable, with appropriately conservative adapters and explicit resource/version checks.
The duplicated representation of claim evidence in graph edges and Claim fields creates
an important correctness defect. The passing tests do not cover this public API path.

## Finding 1 — P1: Graph contradiction edges bypass claim verification safeguards

Location: `reasoning-generalization-tracer/src/rg_tracer/ontology/graph.py:76`,
`graph.py:184`, and `graph.py:201`; the reasoning adapter also creates these edges through
`ontology/adapters.py`'s `_edge` helper.

`add_relation()` accepts `EVIDENCE CONTRADICTS CLAIM` but does not reconcile that edge
with `Claim.contradicting_evidence`. Both `promote_claim()` and `validate()` check only
the Claim evidence lists. Consequently, a claim with an explicit unresolved contradictory
edge can become VERIFIED and export successfully. The reverse order is also exposed:
adding a contradiction edge to a previously verified claim does not invalidate its status.
Direct SUPPORTS edges likewise do not establish the supporting evidence that the promotion
API expects, including edges produced by the reasoning adapter.

This is an actual violation of the conflicting-knowledge lifecycle in sections 34 and 44,
the independent-verification boundary in the combined core rule, and the documented
guarantee that unresolved contradictory evidence prevents promotion.

Targeted probe executed against the actual package:

```python
graph.record_evidence("claim", "support")
graph.add_relation(OntologyRelation("counter", "CONTRADICTS", "claim"))
graph.promote_claim("claim", "passed-task-verification")
graph.validate()
```

The graph contained both the SUPPORTS and CONTRADICTS edges, yet printed `VERIFIED ()`
for the claim's status and contradictory evidence list. The verifier was properly
claim-bound with `method="task_verification"` and `result="passed"`.

Correction: enforce one consistent evidence invariant for every public insertion path.
Either synchronize evidence edges and Claim fields centrally, or derive the promotion and
export checks from all asserted evidence relations as well as direct Claim references.
Preserve atomic edge insertion, earlier status history, and every source observation.

Regression coverage: direct support then direct contradiction before promotion; direct
contradiction after promotion; a reasoning-adapter graph containing both relations;
and preservation of atomic rejection and previous evidence/history.

## Finding 2 — P2 / documentation BLOCK: Cases 14–17 still described as overlay aliases

Location: `reasoning-generalization-tracer/docs/17_case_framework.md:240–244`.

The Migration Note still says cases 14–17 should be treated as "aliases only for
ambiguity-handling overlays". They are canonical V5 cases, not aliases. This contradicts
the document's new introduction and the binding requirement to distinguish behavioral
identity from DSPy's overlay implementation. The text could cause a consumer to omit
four canonical cases from evaluation or reporting.

Violated precision rules: one meaning per term, agreement with canonical sources, and
independently observable requirements. This is a BLOCK because it changes the intended
taxonomy, not merely wording style.

Exact correction:

> Cases 14–17 are canonical V5 ambiguity-handling cases. The classifier selects them
> when callers supply explicit ambiguity mode and stakes metadata. They do not replace
> ordinary IDK cases 9–13. Existing case IDs remain stable.

Verification: review this note against `canonical_case_ids()` and the pinned manifest;
search the audited documentation for the obsolete "aliases only" description.

## Checked requirements and scope limits

The reviewed contract derives canonical names from the single packaged mirror, pins
the requested upstream commit, validates hashes and version, exposes exactly IDs 1–17,
keeps Case 0 non-canonical, and uses registered stripe/subtype/repeat coordinates.
Legacy aliases and nested V3 overlays remain available. Reward implementation changes
are explanatory docstrings; numeric reward components remain separate from identity.
The explicit sync utility is the only new network path and is outside normal runtime.

The ontology uses `rg-ontology-v1` independently of the behavioral and overlay versions.
Typed relation signatures distinguish semantic cycles from execution DAG constraints.
Feature/concept identities remain separate; causal contributions require endpoint-bound
intervention verification. Adapters retain complete JSON source snapshots and unmapped
diagnostics, and canonicalization retains observations under explicit scoped equivalence.
No ontology-shaped-output reward was introduced. Existing specialized schemas remain.

The self-play follow-up normalizes missing/null ontology annotations, rejects malformed
annotations clearly, and validates coordinates before sampler construction/output creation.
Its scope and tests are appropriate; no additional issue was found in that follow-up.

Evidence supplied by the parent: baseline 294 tests passed; implementation suite 376
passed; ontology 38 and combined specialized 69 passed; Black/Ruff and installed-wheel
smoke checks passed; metadata follow-up focused tests 31 passed. This reviewer did not
rerun the whole suite. The independent targeted contradiction probe above reproduced
the uncovered defect. No other concrete important/critical defect or missing mandatory
implementation requirement was found during this review.
