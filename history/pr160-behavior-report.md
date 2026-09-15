# PR 160 behavior fix evidence

## Scope and implementation

The binding specification is `history/pr160-review-plan.md`. This worker changed only
the five Python files assigned by that plan and this report. No staging, commits,
publication, dependency changes or canonical resource modifications were performed.

- `reasoning-generalization-tracer/scripts/check_17case_upstream_sync.py` verifies each
  local resource against its recorded hash, then in network mode compares bytes at
  `upstream_commit` before requesting main. Pinned fetch failures and byte mismatches
  raise `ValueError` naming the pinned commit and resource. Main is still resolved
  once, and both drift comparisons use that resolved commit. Offline mode makes no
  network requests. The four existing status strings remain unchanged.
- `reasoning-generalization-tracer/src/rg_tracer/epistemic_cases/reporting.py` resolves
  nonnull current/canonical/legacy IDs, rejects unequal IDs or noninteger aliases,
  and checks every supplied canonical field against `evaluation_identity`. Missing
  or null legacy IDs allow another nonnull ID or default Case 0. Supplied canonical
  null ID/key/title is valid only for fallback. Canonical ID 0 remains invalid.
- `reasoning-generalization-tracer/src/rg_tracer/schema_v3/case_v3.py` replaces reward
  components with neutral defaults only when confidence is missing and the final
  case is 0. Diagnostic overlays remain available on the result. Observed confidence
  rewards and ambiguity Cases 14–17 keep their existing computations.
- `tests/test_v5_sync.py` and `tests/test_v5_integration.py` contain the regressions.
  Every function and nested function in the five owned Python files now has a
  contract or behavior-specific docstring.

## Before and after

Before production edits, the new regression batch produced **18 failed, 40 passed**:
four network ordering/provenance failures, nine accepted identity conflicts, one
null legacy-ID resolution failure, and four nonneutral missing-confidence rewards.
The correct-answer fallback with all diagnostic bonuses had total 3.2; IDK fallback
had total 2.45. The fixed fallback has every component and total equal to 0.0.

The first implementation produced **58 passed**. Expanded type/range, null canonical
identity and observed-confidence tests produced **88 passed**. The final targeted
run, including both pinned resource failure positions and main errors/version drift,
produced **96 passed in 7.16 seconds**.

Observed-confidence baselines were read by executing `case_v3.py` from commit
`c46c840` in an isolated Python module. Without overlays, aligned correct-answer
high/low totals were 3.0/2.0; unsupported IDK high/low totals were -1.0/1.25.
Tests assert the unchanged decomposed components plus the 1.2 diagnostic bonus
(totals 4.2, 3.2, 0.2, 2.45). Tests for Cases 14–17 compare missing confidence with
the historical 0.0 evaluation surrogate and retain positive overlay components.

## Verification

Run from `reasoning-generalization-tracer` using
`C:/Users/evanh/Documents/Codex/v5env-gepa/Scripts/python.exe`:

```text
python -m pytest tests/test_v5_sync.py tests/test_v5_integration.py -q
python -m black --check --line-length 100 <five owned Python files>
python -m ruff check --select F,E501 --config line-length=100 <five owned Python files>
```

Black and Ruff passed. An AST audit of all functions and nested functions in the
five owned Python files found no missing docstrings. Docstrings specify conditions
and observable results; no unresolved documentation precision finding was identified.

## Limits and handoff

HTTP is mocked at `urlopen`; tests neither contact GitHub nor claim live upstream
availability. The root owns full-suite validation, installed-wheel verification,
combined independent review, Beads updates and publication to the existing PR.
The only reward behavior change is the approved missing-confidence final Case 0
neutralization, including diagnostic bonuses.
