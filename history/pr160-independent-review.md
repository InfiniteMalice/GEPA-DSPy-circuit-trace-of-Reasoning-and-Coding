# PR 160 independent fix review

Verdict: PASS for specification compliance, code quality and documentation precision.
No unresolved findings remain in the approved fix scope.

Reviewed against c46c8404616cee3b916f756f6f90fcf37630b840 and the binding history/pr160-review-plan.md. Scope is the three CodeRabbit behavioral fixes and documentation/docstring changes. Source, index and HEAD were read-only; this report is the only reviewer-written file.

Behavioral findings resolved

Pinned provenance: check() verifies local hashes, then compares each resource against bytes from upstream_commit before fetching main. A mismatch or OSError names the pinned commit and resource; main cannot conceal invalid provenance. Offline directory mode does not enter the network branch. Regression fixtures cover both resource positions, pinned mismatch/read errors, main read errors, drift and incompatible versions.

Report identity: all nonnull case_id, canonical_case_id and reward_case inputs must be actual equal integers. evaluation_identity validates range and coordinates; supplied canonical ID/key/title/flag then require exact type and value equality. This catches ignored aliases and labels while retaining omitted IDs, nullable legacy fields and null canonical fallback identity. Canonical ID 0 cannot silently become valid canonical identity.

Reward fallback: neutralization happens after ambiguity routing, and only when confidence is None and final case_id is 0. Replacing the complete RewardComponents object zeros base components, all diagnostic bonuses and total. Diagnostic overlays remain attached. Observed-confidence and Cases 14–17 use the existing computations. Regression evidence includes before/after failures, nonzero overlays, four observed-confidence baselines and all four ambiguity routes. The reward exception is explicitly authorized in the binding plan. No broader reward changes or additional authority were introduced.

Documentation findings resolved

The first review identified three BLOCK findings: the package README generalized neutral rewards to every V3 Case 0; epistemic_alignment.md omitted the final answer/IDK condition and classifier actor; the schema_v3 README used metadata for both identity-independent overlays and identity-selecting ambiguity inputs. The final text distinguishes historical base rewards from the missing-confidence V3 exception, names classify_case_v3 and the ambiguity exception, and separates research overlays/coordinates from explicit ambiguity inputs. A final absolute statement was corrected to say ambiguity inputs can select Cases 14–17, preserving ordinary routing for low-stakes proceeding and epistemic abstention.

Three WARN findings are resolved: TRMSampler.generate now documents k candidates and output fields; two schema tests identify the terminology/authority boundaries and concrete registry/example/signature checks instead of unspecified requirements. Other concise test docstrings state asserted behavior. The V5 migration and sync descriptions agree with the implementation and regression conditions. No documentation BLOCK or WARN remains from this scoped review.

The original docstring coverage warning is addressed without executable changes. An independent audit of all 23 original PR Python files found 255 class/function definitions with 255 docstrings and no missing definitions, including nested functions. Useful production contracts describe validation, output, side effects or compatibility behavior.

Verification and limits

Independently stripped docstrings and compared executable ASTs for all 13 documentation-only Python files with actual diffs: equal to c46c840. Five additional documentation-owned files have no diff. Repeated AST comparison for self_play.py and test_schema_v3.py after the wording repair: equal.

A current-source spot check reproduced the documentation boundary: observed confidence 0.9 with no expected answer and an O5 provenance overlay yields Case 0 with total 0.65. This confirms why the final docs distinguish missing-confidence neutralization from other fallback results; it is preserved behavior outside the authorized reward exception.

Regression source and history/pr160-behavior-report.md were reviewed, including reported 18 before-fix failures and 96 targeted passes. This reviewer did not rerun the integration suite. Root reports 466 full-suite tests passed; its conftest prepends the current checkout source. Root owns the final wheel and publication checks. The concrete reviewer spot check explicitly prepended current src because the environment editable install otherwise resolves an older worktree. Network regression tests mock HTTP and do not certify upstream availability.

Review is limited to this fix batch; the earlier implementation review and unrelated baseline follow-ups remain separate.
