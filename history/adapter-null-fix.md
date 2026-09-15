# Nullable ontology adapter labels

Scope: CodeRabbit's verified nullable node kind/type finding and the adjacent nullable
relation normalization paths. No broad schema changes, commits, staging, or subagents.

Changed `reasoning-generalization-tracer/src/rg_tracer/ontology/adapters.py` and
`reasoning-generalization-tracer/tests/test_ontology_adapters.py`.

Raw mappings with `kind: null` now preserve the existing `REPRESENTATION` fallback.
Raw attribution nodes with `type: null` now preserve the `MODEL_FEATURE` fallback.
Both adapters also contained the same `.upper()` crash for `relation: null`; those
relations now remain unmapped diagnostic entries. An omitted attribution relation
still defaults to `CONTRIBUTES_TO`, as before. All source snapshots retain the original
null fields and payloads.

Test-first evidence: four new parameterized cases reproduced four AttributeErrors
before the fix. They check fallback types, unmapped relations, snapshot equality, and
graph validity. After the fix:

- Combined ontology/reasoning/attribution focused tests: **77 passed in 4.99 seconds**.
- Adapter suite after formatting: **28 passed**.
- Black `--check --line-length 100` on both changed Python files: unchanged, exit 0.
- Ruff `--select F,E501 --line-length 100` on both changed Python files: passed, exit 0.

Tests ran with `PYTHONPATH=src` and the shared `v5env-gepa` Python interpreter.
No other nullable schema fields were broadened in this scoped fix.
