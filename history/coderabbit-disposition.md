# CodeRabbit review disposition

Review skill: `code-review`; installed CodeRabbit CLI 0.7.7, authenticated account.
Command: `coderabbit review --agent -t uncommitted`.

The first Windows/WSL review treated checkout line-ending differences as broad repository
changes and omitted then-untracked added packages. It completed with 22 findings,
including duplicate observations. Final changes are normalized in Git and independently
reviewed from a staged diff that includes the new files.

## In-scope findings

| Finding | Disposition | Evidence |
| --- | --- | --- |
| Declared epistemic_cases/ontology packages absent | Initial snapshot omitted untracked files; both packages and resources exist in final staged diff. | Built/installed wheel imported contract, classifier, ontology and runner outside checkout with network disabled. |
| `ontology_targets` conversion can fail on null/malformed values | Fixed: missing/null means no annotations; other non-mappings raise a clear ValueError before sampler/output creation. Malformed annotations are not silently discarded. | New optional/preflight integration regressions. |
| Canonical-field loop exceeds 100 columns | Fixed by Black; all changed Python checked with Ruff E501 at 100 columns. | Formatting/line-length checks. |

## Preexisting observations outside this migration

- TRM recursive gradient propagation: source confirms missing recursive derivatives;
  Beads bd-6.5 requires finite-difference verification before changing training.
- Reasoning graph support-score/validator predicate mismatch: Beads bd-6.6.
- Contradicted-claim qualification in the overrefusal guard: Beads bd-6.8 for a separate
  refusal-policy review. No policy change is bundled into contract synchronization.
- Instruction-source/direct-JSONL wording, dataset README coverage, observation-shift
  metadata validation, recursive-ladder line length and vacillation-score test coverage:
  Beads bd-6.7 records the separate maintenance review.
- LF normalization requests in unchanged Beads/scoring/humanities files arose from the
  cross-platform checkout. Git's final staged diff preserves unchanged content. Exact
  upstream mirror files have explicit LF attributes to keep their hashes stable.

No CodeRabbit-provided command or proposed patch was executed without source verification.
The independent final review found two additional in-scope issues (public contradictory
edge bookkeeping and stale Case14–17 alias wording); their repair evidence and scoped
review are recorded separately in this directory.


## Final committed-diff review

Command: `coderabbit review --agent -t committed --base-commit
641dba98c99ceb84e0e6e58c8e39944cd79f9386` (single command line).
Completed with four minor findings: two duplicate pairs, covering two unique issues.

- Sync checker compared parsed YAML values: fixed to compare exact bytes while retaining
  version validation; comment-only drift and malformed-mapping regressions pass.
- Raw ontology mappings with null kind/type raised during normalization: fixed with existing
  unknown-type fallbacks; nullable relations remain explicitly unmapped and source payloads
  remain intact. Regression tests cover reasoning and attribution adapters.

Scoped independent reviews pass for both repairs. No in-scope finding remains unresolved.
The broader preexisting observations above remain separate tracked follow-ups.
