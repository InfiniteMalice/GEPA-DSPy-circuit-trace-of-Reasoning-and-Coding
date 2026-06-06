# Semantic Constraint Synthesis

RG-Tracer includes a shadow-first semantic constraint compiler for bounded
synthetic finite-domain tasks. The supported flow is:

```text
synthetic natural-language requirement
  -> rule-based compiler
  -> provenance-tracked candidate constraints
  -> verifier
  -> explicit finite-domain lattice
  -> shadow, advisory, or toy-only gated projection
```

The default compiler supports only explicit toy clauses such as:

- `the answer must be even`
- `the answer must be greater than 2`
- `the answer must be less than or equal to 5`
- `the answer must not be 3`
- `the answer must be one of 2, 4, 6`

Unsupported fragments, ambiguity, duplicates, contradictions, confidence
bounds, provenance, and candidate-set emptiness are surfaced in structured
metadata. Only verified constraints may be converted into
`DeductionConstraint` records, and `gated_toy_only` is limited to synthetic
finite-domain tasks.

The compiler does not execute generated code, control robots or spacecraft,
modify external systems, or treat open-ended language as verified. Recovered
embedding-space lattices remain shadow-only.

Reference: [Semantic Constraint Synthesis for Adaptive Trajectory Optimization via Large Language Models](https://arxiv.org/abs/2606.04123).
