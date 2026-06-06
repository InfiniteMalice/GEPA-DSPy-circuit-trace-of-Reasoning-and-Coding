# Explicit Lattice Deduction

The LDT-inspired projector is a small explicit abstract domain for toy tasks. It represents a finite
candidate set as a lattice element and uses subset inclusion as the partial order.

## Operations

- `meet(left, right)`: candidate-set intersection.
- `join(left, right)`: candidate-set union.
- `is_below(left, right)`: subset inclusion.
- `canonicalize(element)`: stable representation for equivalent candidate sets.
- `project(element, constraint)`: intersection with an explicit allowed-candidate constraint.

Empty candidate sets indicate contradiction. Singleton candidate sets indicate resolution.
Multi-candidate sets remain unresolved. All diagnostics are public and JSON serializable.

## Adapters

The initial adapters cover `addition`, `parity`, and `finite_domain_constraint`. The finite-domain
dataset includes unique resolution, unresolved states, contradiction, irrelevant constraints,
duplicate constraints, order-invariant equivalent constraints, early resolution, and projection
budget exhaustion.

## Modes

- `off`: projection is not invoked.
- `shadow`: diagnostics are recorded but do not alter output.
- `advisory`: diagnostics can recommend a candidate or abstention without overriding hard gates.
- `gated`: explicit task-local projection can constrain final output. Contradiction or unresolved
  states trigger abstention when configured.

Recovered embedding-space lattices remain shadow-only. The repository includes protocol interfaces
for future representation probes, but no fake embedding recovery and no recovered-lattice gating.
## Semantic Constraint Bridge

`rg_tracer.semantic_constraints` adds a bounded rule-based bridge from
synthetic finite-domain requirements into explicit `DeductionConstraint`
records. The bridge verifies supported grammar, provenance, confidence,
duplicates, contradictions, and candidate non-emptiness before constraints can
enter a task-local lattice. `shadow` and `advisory` modes report metadata;
`gated_toy_only` is restricted to verified synthetic finite-domain tasks.
Recovered or inferred representation lattices remain shadow-only.
