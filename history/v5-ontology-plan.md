# V5 contract and reasoning ontology implementation plan

Spec: the two user supplied requests, sections 0–61. Preserve all requested behavioral,
provenance, compatibility, graph and verification boundaries.

## Global constraints

Canonical behavior is 17case-v5, IDs 1–17; Case 0 is non-canonical. Mirror upstream
commit a12d8d27134bc38b5aa43aff33b84db2f7f51bce without runtime network access.
Keep V3 overlays and reward policy. Use rg-ontology-v1 independently of V5.
Python lines have a 100-character limit. Beads is the only task tracker.
New registries ship as package data; installed wheels must work outside this checkout.

## Design

Use an importlib.resources-backed epistemic_cases package with exact upstream YAML,
hashes and provenance, immutable case records, strict case/stripe/repeat validation,
legacy aliases and an explicit read-only upstream comparison script. CaseV3Result
normalizes names and serializes canonical identity and provenance. Ambiguity routing
must classify low-stakes assumptive proceeding using answer evidence, and loops as 17.

Ontology uses a small machine-readable registry, common typed entities, claims,
transformations and relations. Graph views separate semantic cycles from provenance
DAGs. Adapters preserve original metadata and identities without replacing source
graphs. Conservative equivalence records retain each observation's provenance.
Mechanistic evidence cannot establish semantic truth or justify causal edges without
intervention evidence. No ontology rewards.

## Task 1: Contract and classification

Create src/rg_tracer/epistemic_cases and pinned package-data YAML; update schema_v3/case_v3.py,
tests/test_schema_v3.py, and new tests/test_epistemic_cases.py. API: get_case(id),
get_case_key(id), get_case_title(id), get_expected_behavior(id),
get_confidence_semantics(id), get_stakes_semantics(id), canonical_case_ids(),
is_canonical_case(id), resolve_legacy_case_name(name), validate_coordinate(case_id,
stripe='NONE', stripe_subtype=None, repeat_id=0), contract_provenance().
Test exact pinned contract fields/hashes, all 17 classifier outcomes, legacy aliases,
invalid IDs/coordinates, fallback serialization and unchanged numeric rewards.

## Task 2: Ontology core and adapters

Create src/rg_tracer/ontology, tests/test_ontology.py and docs/ONTOLOGY.md.
Implement user sections 29–60: registry, typed entities/relations, provenance,
epistemic statuses, four graph views, conservative canonicalization, transformations,
ReasoningGraph/ConceptLatticeSpec/AttributionGraph/SemanticTag/reasoning-unit adapters,
V5 case/stripe adapters and failure/repair/regression representation.
Consume the Task 1 API above without duplicating V5 identities.
Test lossless adapters, semantic cycles, forbidden provenance cycles, conflicting
evidence retention, constraints preventing false equivalence, and mechanistic boundary.

## Task 3: Runtime integration and documentation

Update package metadata, runner records and aggregate reporting, synthetic examples,
root/package README, docs/17_case_framework.md, docs/schema_v3.md and
docs/epistemic_alignment.md. Carry CASE × STRIPE × REPEAT and pinned provenance into
new outputs; report Case 0 separately. Audit all direct thought-reward paths and file
a Beads follow-up without changing weights. Document legacy mappings and version axes.

## Verification and review

Run baseline suite before implementation, focused new tests and existing specialized
graph/classifier/reward tests, then full repository suite. Run Black, unused-import
checks and line-length checks for changed Python. Build a wheel and smoke-test resources
outside the checkout. Review spec compliance and code quality, including exact upstream
parity, no stale canonical tables and precise docs. Record actual results and limitations.

## Workflow rulings

The workspace is a fresh isolated clone on a feature branch; another worktree adds no
isolation. Beads CLI/MCP is absent and no bd process is running; use the AGENTS.md
documented direct JSONL fallback and log seed status. Permanent architecture rationale
will live in docs/adr; this implementation plan remains in history per repository rules.
Task 2 is delegated while the controller implements Tasks 1/3; file ownership is disjoint.
