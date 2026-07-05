# Graph-Native Reasoning

## Summary

RG-Tracer now includes optional public reasoning graph metadata helpers.

## Reference

Graph-native reasoning follows Graph-PRefLexOR (Pal et al., 2026; arXiv:2607.00924).

## Why It Belongs

RG-Tracer already studies attribution graphs, verification, and reasoning generalization. Public
`graph_json` records give those workflows a traceable concept/relation layer without depending on
hidden chain-of-thought.

## Implemented

- `rg_tracer.reasoning_graphs.schema` defines node, edge, and graph dataclasses.
- `validators` checks unique node IDs, edge endpoints, support paths, cycles, and contradictions.
- `graph_score` returns auxiliary diagnostics and scores.
- `phase_parser` extracts public structured sections including `<graph_json>`.

## Future Work

Adapters can later blend graph scores into GEPA scoring, transfer evaluations, or attribution
metrics. They are not wired into default CLI behavior.

## Compatibility Notes

Graph-native reasoning is public structured metadata, not hidden thought extraction. Graph scoring
is optional and does not replace GEPA rubric scoring.
