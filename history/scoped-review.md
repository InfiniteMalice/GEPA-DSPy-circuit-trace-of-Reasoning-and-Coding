# Scoped independent re-review

Scope: `work/review-fixes.diff`, `history/final-fix-report.md`, and the corresponding
updated source/tests. This review checks disposition of the original two findings,
the public canonical-identity constructor fix, and the runner metadata preflight.
It does not reopen unrelated source or reward-policy work.

## Final review verdicts

**Spec compliance: PASS.** Both blocking findings from `history/final-review.md`
are addressed. The reviewed fixes retain canonical V5 case identity, independent
ontology/overlay versions, conservative equivalence, and the semantic/mechanistic boundary.

**Code quality: PASS.** Evidence bookkeeping now has one insertion path, constructor
invariants are shared, and invalid runner metadata fails before sampling/output creation.
No new important/critical defect was found in the fix diff. No documentation BLOCK remains
from this review. Final integrated test completion remains with the parent.

## Dispositions

### Finding 1: graph evidence bypass — ADDRESSED

`OntologyGraph.add_relation()` validates the relation, computes complete claim updates,
then appends the edge and installs those updates. No graph mutation occurs before
validation/update construction succeeds. `record_evidence()` delegates to this same path.

Direct EVIDENCE SUPPORTS/CONTRADICTS CLAIM edges populate deduplicated Claim evidence
references. A contradiction changes status to CONTRADICTED, preserving support, provenance,
verification references, and the prior status. An edge after promotion therefore cannot
leave the current claim VERIFIED. Claim-to-Claim contradictions dispute both endpoints
without treating either claim as task-level Evidence.

Both promotion and strong-status export independently inspect asserted contradiction
edges, protecting against stale cached Claim fields. The reasoning adapter uses this
central insertion API, so its source evidence relations follow the same invariant.

Inspected regressions cover direct support promotion; Evidence contradiction before/after
promotion; both Claim-contradiction directions; stale verified export; invalid-edge
atomicity; and a reasoning-adapter graph containing support and contradiction. These
exercise the original failure and relevant neighboring insertion paths.

### Finding 2: canonical cases described as overlay aliases — ADDRESSED

`docs/17_case_framework.md` now explicitly calls cases 14–17 canonical V5 ambiguity-handling
cases and distinguishes them from ordinary IDK cases 9–13. The original "aliases only"
wording is absent. The replacement agrees with the pinned manifest and the document's
introduction. The changed ontology documentation also matches the new evidence behavior.

### Public CanonicalIdentity constructor invariants — ADDRESSED

`CanonicalIdentity.__post_init__()` converts the observation input to an immutable tuple
and runs the shared validator. The factory delegates to that constructor. Both entry points
therefore require a non-empty identity/key, at least two distinct Claim observations,
matching semantic keys and scoped constraints, and a passed observation-bound equivalence
verification. Individual observations and provenance remain intact. Inspected negative tests
cover identity, observation count/duplication, key/constraint mismatch, and verifier binding/
result, plus defensive copying of the input sequence.

### Runner null annotations and coordinate preflight — ADDRESSED

`run_self_play()` validates stripe/subtype/repeat immediately after loading the problem.
Missing/null ontology targets normalize to an empty mapping; non-mapping values raise a
specific ValueError before constructing a sampler or creating output directories.
Candidate records use the validated mapping. The change leaves reward behavior intact.
Inspected regressions cover null and populated annotations, invalid annotations, invalid
stripe/repeat, and absence of output creation on failure.

## Verification evidence

This reviewer inspected the actual final source, fix diff, documentation, and regression
tests. A read-only search confirmed removal of the obsolete alias wording. No concrete new
suspected defect justified repeating tests, so no test suite was rerun during this scoped
review.

Reported execution evidence from the implementer/parent: ontology suite 55 passed; combined
ontology and specialized compatibility suite 86 passed; Black with the 100-column limit
and Ruff F/E501 passed. The earlier runner metadata-focused suite had 31 passing tests.
The parent is running the final integrated suite and owns final packaging verification.

This scoped PASS supersedes the pending-fix verdicts in `history/final-review.md` for the
reviewed final source.
