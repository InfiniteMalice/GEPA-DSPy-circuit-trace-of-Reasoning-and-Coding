# Ontology implementation report

Task: `history/v5-ontology-plan.md`, Task 2; user ontology specification sections 29–61.
The parent controls contract implementation, packaging, runner integration, Beads, and commits.
This bounded implementation changes only `reasoning-generalization-tracer/src/rg_tracer/ontology`,
`tests/test_ontology*.py`, `docs/ONTOLOGY.md`, and this report.

## Outcome

Implemented the independent `rg-ontology-v1` registry with 36 entity types and 50 canonical
relation types. Runtime enums derive from packaged `registry.json`. Expanded relation metadata
declares domain/range, direction, inverse, transitivity, symmetry, and per-view cycle rules.
Existing graph schemas are preserved; adapters provide typed projections plus full source
snapshots and unmapped-data diagnostics. No ontology reward or scoring change was introduced.

## Entity types

```text
TASK OBJECTIVE CONSTRAINT ASSUMPTION CLAIM EVIDENCE CONCEPT REASONING_UNIT STEP
TRANSFORMATION REPRESENTATION ACTION TOOL_ACTION ARTIFACT OBSERVATION OUTCOME
VERIFICATION EVALUATION_RECORD CANONICAL_CASE ROBUSTNESS_STRIPE STRIPE_SUBTYPE REPEAT
FAILURE FAILURE_FAMILY REPAIR REGRESSION_TEST MODEL MODEL_FEATURE CIRCUIT_FEATURE
ACTIVATION_PATTERN ATTRIBUTION_EDGE ATTRIBUTION_PATH CONCEPT_PROBE
MECHANISTIC_OBSERVATION RUN DATASET_EXAMPLE
```

Claims and transformations have richer frozen records. All entities share stable identity,
aliases, metadata, and provenance. Lattice attributes are concept properties, not a second
vocabulary. Generalization levels and failure labels are diagnostic registry values without
new evaluation scoring.

## Relation types

```text
INSTANCE_OF SPECIALIZES GENERALIZES EQUIVALENT_TO PARAPHRASE_OF REPRESENTATION_OF
HAS_OBJECTIVE REQUIRES CONSTRAINED_BY SUPPORTS CONTRADICTS ENTAILS DEPENDS_ON
DERIVED_FROM CONTAINS PRODUCES PROVIDES PRECEDES VERIFIED_BY VERIFIES OPERATES_ON
TRANSFORMS TRANSFORMS_TO PRESERVES CHANGES BREAKS_INVARIANT USES COMPOSES_WITH
GENERALIZES_TO ASSOCIATED_WITH CANDIDATE_ENCODING_OF PREDICTIVE_OF
CAUSALLY_CONTRIBUTES_TO CONTRIBUTES_TO ATTRIBUTED_TO HAS_FEATURE ACTIVATED_IN
ACTIVATED_DURING TESTS SUPPORTS_MAPPING HAS_CASE HAS_STRIPE HAS_SUBTYPE HAS_REPEAT
OBSERVED_IN TRIGGERED_BY FAILS_UNDER FAILS_WITH REPAIRED_BY REGRESSION_OF
```

Three explicit input aliases normalize to canonical meanings: `INSTANCE_OF_FAILURE` →
`INSTANCE_OF`, `EXPRESSES` → `REPRESENTATION_OF`, `QUALIFIED_BY` → `CONSTRAINED_BY`.
Relation-definition lookups return defensive copies so callers cannot mutate the causal gate.

## Public API

`rg_tracer.ontology` exports `ONTOLOGY_VERSION`, `load_registry`, `EntityType`, `RelationType`,
`EpistemicStatus`, `GraphView`, `OntologyEntity`, `Provenance`, `Claim`, `Transformation`,
`OntologyRelation`, `OntologyGraph`, `CanonicalIdentity`, `candidate_equivalences`, and
`canonicalize_claims`.

`OntologyGraph` exposes `add_entity`, `add_relation`, `record_evidence`, `promote_claim`,
`validate`, and `to_dict`, plus read-only entities and relations. Failed edge additions and
failure/repair additions are atomic. Strong claim statuses require source evidence and a
passed claim-bound task verification; contradictory evidence and status history are retained.

`rg_tracer.ontology.adapters` supplies:

- `adapt_reasoning_graph`: original node IDs, task, answer reference, text, relations, weights,
  and metadata survive. Known node kinds map to semantic types; unknown kinds become
  representations. Unsupported relation semantics remain in the snapshot and diagnostics.
- `adapt_concept_lattice`: concepts/attribute properties and provisional implications preserve
  domain and `shadow_only`; no world-knowledge authority is inferred.
- `adapt_attribution_graph`: model/circuit features and attribution edges preserve model,
  task, layer, activation, signed attribution, token position(s), phase, and extra raw fields.
- `adapt_semantic_tag`: existing error values become failure families; positive `SUPPORTED`
  and `ENTAILED` tags remain validation concepts. No incompatible duplicate taxonomy.
- `adapt_reasoning_units`: registry families become canonical concepts; reasoning-unit records
  preserve definitions and link known dependencies/composition partners. Unknown partners are
  reported instead of invented.
- `adapt_transformation`: records inputs/outputs, preserved/changed properties, expected and
  observed invariants, symmetry breaks, inverse, composition parent, and complete group overlay.

Graph adapters return `AdapterResult(graph, unmapped)` with a defensive JSON-compatible
`source_payload`. Tuple-valued dataclass fields normalize to JSON arrays. These functions do
not mutate source structures or claim round-trip preservation of Python container classes.

`rg_tracer.ontology.evaluation` supplies:

- `adapt_evaluation`: consumes the shared V5 contract API; external case/stripe/subtype IDs,
  titles, and provenance come from that contract. Supplied metadata is checked for inconsistent
  version, canonical values/types, provenance, and case name. Legacy absent fields are accepted.
  Case 0 remains operational metadata with no canonical case entity or HAS_CASE edge.
- `add_failure_repair`: atomically adds failure/evaluation, repair/retest, and regression links
  while retaining the original failure. A retest link identifies the checking evaluation;
  callers must inspect its result before treating a repair as successful.

## Graph invariants and epistemic boundary

Semantic, evaluation, and mechanistic views allow cycles. Execution/provenance requires a DAG
after normalizing relation direction: DERIVED_FROM points toward a source, so its direction
is reversed for source-before-result cycle detection. Semantic dependencies do not enter DAG
validation. Every edge checks its registered per-view domain and range.

Mechanistic observations cannot use semantic SUPPORTS or certify task truth. Features cannot
be equated with semantic concepts. A causal edge requires a passed intervention/causal-test
verification referring to the exact source and target. A task verifier cannot certify a
mechanistic mapping; attribution magnitudes do not silently become causal tests.

Canonicalization is conservative: explicit shared semantic keys and exact scoped constraints
produce candidates. Canonical grouping additionally requires a passed equivalence check naming
exactly the observations. Each original claim, provenance, evidence list, and epistemic status
survives. Different authorization/domain constraints prevent surface-based identity merges.

## Verification performed

Test-first evidence:

1. Core tests initially failed collection because the ontology package did not exist.
2. Adapter tests initially failed collection because the adapter module did not exist.
3. Direct stripe/Case-0 edge bypass regression tests failed before validation was added.
4. Defensive relation metadata and supplied V5 identity tests failed before review fixes.

Final focused ontology result: **38 passed**.

Final combined ontology and specialized compatibility result: **69 passed in 4.37 seconds**:

```text
PYTHONPATH=src python -m pytest tests/test_ontology.py tests/test_ontology_adapters.py
  tests/test_reasoning_graphs.py tests/test_concept_lattice_registry.py
  tests/test_lattice_core.py tests/test_lattice_adapters.py
  tests/test_attribution_metrics.py tests/test_attribution_integration.py tests/test_semantics.py -q
```

The actual Windows run set `$env:PYTHONPATH='src'` and used
`C:/Users/evanh/Documents/Codex/v5env-gepa/Scripts/python.exe`.

Final formatting/static checks:

- `python -m black --check --line-length 100 src/rg_tracer/ontology tests/test_ontology.py
  tests/test_ontology_adapters.py`: nine files unchanged, exit 0.
- `python -m ruff check --select F,E501 --line-length 100 src/rg_tracer/ontology
  tests/test_ontology.py tests/test_ontology_adapters.py`: all checks passed, exit 0.
- Registry import smoke: `36` entity types, `50` relation types.

An initial Ruff invocation accidentally used its default 88-column limit; it reported lines
between 89 and 100 characters. The corrected invocation explicitly uses the repository's
100-column limit and passes. No unused-import errors remained.

The parent owns full-suite verification and installed-wheel resource checks. This agent made
no commits, staged no files, and changed no specialized source schema or reward implementation.

## Documentation and quality gate

Added `reasoning-generalization-tracer/docs/ONTOLOGY.md`, covering all fourteen requested
topics, a compact architecture diagram, API semantics, V5 namespaces, diagnostics, and limits.
Reviewed against the task specification and documentation precision gate. No unresolved
documentation BLOCK findings. Runtime requirements described in the document correspond to
the focused tests; future capabilities are explicitly identified as future work.

## Implemented, experimental, and deferred

Implemented: registry and schema validation, four graph views, lossless JSON source snapshots,
provenance retention, conflicting evidence, canonical identity grouping, transformation records,
all requested source adapters, V5 identity validation, failure/repair/regression representation.

Experimental/research assertions: semantic keys, equivalence verification results, concept
implications, candidate feature/concept mappings, and reported causal-test validity. The library
checks structure and bound references; it cannot establish the empirical reliability of the
upstream evaluator or omitted constraints.

Deferred: automatic semantic-key discovery, automatic cross-run feature identity resolution,
graph query engine, causal experiment execution, verifier reliability assessment, and automatic
generalization-failure classification. No future capability is represented as implemented.
Higher abstraction and ontology-shaped output receive no intrinsic reward.
