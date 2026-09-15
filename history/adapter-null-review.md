# Scoped nullable-adapter review

Scope: the current uncommitted changes to `ontology/adapters.py` and
`tests/test_ontology_adapters.py`, plus `history/adapter-null-fix.md`.

**Nullable adapter finding: ADDRESSED. Spec compliance: PASS. Code quality: PASS.**

The normalization now handles explicit null node labels before case conversion.
Reasoning nodes with null `kind` use the existing REPRESENTATION fallback and retain
the unknown-kind diagnostic. Attribution nodes with null `type` use the existing
MODEL_FEATURE fallback. These fallbacks do not promote a claim or assert a mechanistic
explanation. Recognized non-null string labels retain their earlier mappings.

Explicit null relations normalize to an unregistered empty label and remain unmapped
diagnostics in both adapters. No ontology edge or fabricated relation is introduced.
The important attribution distinction is preserved: an omitted relation retains the
existing CONTRIBUTES_TO default, while an explicitly null relation remains unknown.
The source snapshots and source-node/edge metadata retain the original null values.

The four parameterized regressions cover each adapter's nullable node and relation
path, verifying fallback types, lack of invented relations, unmapped diagnostics,
source-payload equality, original null preservation, and graph validity. The changes
do not alter evidence bookkeeping, canonicalization, causal gates, or scoring policy.

No new important defect was found in the scoped diff. The implementation is a small
local normalization fix and does not require broader schema changes.

Verification: independent read-only diff, source, regression, and implementation-report
inspection. The implementer reports four AttributeError failures before the fix,
77 combined focused tests passing afterward, 28 adapter tests passing after formatting,
and passing Black (100 columns) and Ruff F/E501 checks. No broader tests were rerun by
this reviewer; the parent owns the final full-suite and ontology verification.
