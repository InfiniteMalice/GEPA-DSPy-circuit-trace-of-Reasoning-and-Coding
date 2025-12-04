# Beads Database (Manual Log)

Installer status: `curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash`
continues to fail with HTTP 403 due to a proxy block. Installation remains pending until network
access is restored.

## Standards to Seed
- Python line length: 100 characters; follow the import ordering (stdlib, third-party DSPy, local).
- Avoid unused imports and format with `black --line-length 100 .`.
- Keep `.beads/instructions.md` mirrored with `AGENTS.md`; update `AGENTS.md` first.
- Use bd for all issue tracking with `--json`, commit `.beads/issues.jsonl` alongside code, and
  store AI planning docs in `history/` to keep the root clean.

## Repository Snapshot
- Project root README highlights RG-Tracer as the main package and CLI under `reasoning-generalization-tracer/`.
- Key components (from the package README): scoring rubric in `src/rg_tracer/scoring/`, runners in
  `src/rg_tracer/runners/`, concept rewards in `src/rg_tracer/concepts/`, abstention helpers in
  `src/rg_tracer/abstention/`, TRM baseline in `src/rg_tracer/trm_baseline/`, and tests in `tests/`.
- Expected artifacts from runs: `scores.jsonl`, `semantics.jsonl`, `summary.md`, `best.json`, and
  attribution outputs in `runs/<timestamp>/attr/` with metrics in `attr_metrics.jsonl`.
- Default commands available after editable install: `rg-tracer self-play`, `rg-tracer eval`, and
  `rg-tracer trace`, plus humanities and fallback pipelines.

## Next Steps
- Retry the official installer when GitHub access is available, then import these entries into the
  beads database.
- Expand the snapshot with dataset coverage and configuration defaults once beads tooling is live.
- Add bd issues to `.beads/issues.jsonl` once installation succeeds to capture open tasks, blockers,
  and discovered-from links for RG-Tracer work.
