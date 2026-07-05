# Skill Organization And Supply Chain

## Summary

Skill metadata support records manifests, explicit dependencies, and simple risk warnings.

## Reference

Skill organization follows SkillJuror (Chen et al., 2026; arXiv:2606.11543). Dependency risk
follows Skills Are Not Islands (Jia et al., 2026; arXiv:2607.01136). Validated skill discovery is
inspired by ASPIRE (Lu et al., 2026; arXiv:2607.00272).

## Why It Belongs

If RG-Tracer grows reusable reasoning or coding skills, manifests help preserve provenance and make
dependency risk visible.

## Implemented

- `SkillManifest` and `SkillDependency` dataclasses.
- Manifest JSON round trips.
- Dependency graph extraction from explicit manifests.
- Risk warnings for missing version, missing provenance, external services, recursive skill
  dependencies, and dependencies without source.

## Future Work

Natural-language dependency extraction is intentionally left for future adapters.

## Compatibility Notes

Skill manifests are supply-chain metadata, not proof of safety.
