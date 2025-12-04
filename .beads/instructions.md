# Beads Instructions

These instructions mirror `AGENTS.md` so beads can track repository rules. Update
`AGENTS.md` first and copy changes here to keep both files in sync.

## GEPA-DSPy Circuit Tracing Rules
- Python files use a 100-character line limit. No exceptions.
- Imports must be organized as:
  ```python
  # Standard library
  import json
  from typing import Any, Dict, List

  # Third-party (DSPy together)
  import dspy
  from dspy import ChainOfThought as CoT
  from dspy import Signature as Sig
  import numpy as np

  # Local
  from .tracer import CircuitTracer
  ```
- No unused imports. Verify every DSPy module imported is used.
- Black formatting with `black --line-length 100 .`.

## DSPy-Specific Guidance
- Use short signature names (e.g., `class ReasonSig(Sig):`).
- Instantiate DSPy modules with `CoT` or other modules after calling `super().__init__()`.
- Break long argument lists across lines to stay under 100 characters.
- Prefer aliases and shorter names over exceeding the line limit.

## Pre-Commit Checklist
- [ ] Python lines ≤ 100 characters.
- [ ] DSPy imports aliased and organized.
- [ ] No unused signatures or modules.
- [ ] Black formatted.

## Common Mistakes
- Missing `super().__init__()` in `dspy.Module` subclasses.
- Long signature field descriptions that exceed the line limit.
- Forgetting to break long `predict()` or similar calls across multiple lines.
- Importing entire `dspy.teleprompt` modules instead of specific classes.

## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown
TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Auto-syncs to JSONL for version control
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**
```bash
bd ready --json
```

**Create new issues:**
```bash
bd create "Issue title" -t bug|feature|task -p 0-4 --json
bd create "Issue title" -p 1 --deps discovered-from:bd-123 --json
bd create "Subtask" --parent <epic-id> --json  # Hierarchical subtask (gets ID like epic-id.1)
```

**Claim and update:**
```bash
bd update bd-42 --status in_progress --json
bd update bd-42 --priority 1 --json
```

**Complete work:**
```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues.
2. **Claim your task**: `bd update <id> --status in_progress`.
3. **Work on it**: Implement, test, document.
4. **Discover new work?** Create linked issues with
   `bd create "Found bug" -p 1 --deps discovered-from:<parent-id> --json`.
5. **Complete**: `bd close <id> --reason "Done" --json`.
6. **Commit together**: Always commit the `.beads/issues.jsonl` file together with code changes
   so issue state stays in sync with code state.

### Auto-Sync

bd automatically syncs with git:
- Exports to `.beads/issues.jsonl` after changes (5s debounce).
- Imports from JSONL when newer (e.g., after `git pull`).
- No manual export/import needed!

### GitHub Copilot Integration

If using GitHub Copilot, also create `.github/copilot-instructions.md` for automatic instruction
loading. Run `bd onboard` to get the content, or see step 2 of the onboard instructions.

### MCP Server (Recommended)

If using Claude or MCP-compatible clients, install the beads MCP server:

```bash
pip install beads-mcp
```

Add to MCP config (e.g., `~/.config/claude/config.json`):
```json
{
  "beads": {
    "command": "beads-mcp",
    "args": []
  }
}
```

Then use `mcp__beads__*` functions instead of CLI commands.

### Managing AI-Generated Planning Documents

AI assistants often create planning and design documents during development:
- PLAN.md, IMPLEMENTATION.md, ARCHITECTURE.md
- DESIGN.md, CODEBASE_SUMMARY.md, INTEGRATION_PLAN.md
- TESTING_GUIDE.md, TECHNICAL_DESIGN.md, and similar files

### Best Practice: Use a dedicated directory for these ephemeral files

**Recommended approach:**
- Create a `history/` directory in the project root.
- Store ALL AI-generated planning/design docs in `history/`.
- Keep the repository root clean and focused on permanent project files.
- Only access `history/` when explicitly asked to review past planning.

**Example .gitignore entry (optional):**
```gitignore
# AI planning documents (ephemeral)
history/
```

**Benefits:**
- ✅ Clean repository root
- ✅ Clear separation between ephemeral and permanent documentation
- ✅ Easy to exclude from version control if desired
- ✅ Preserves planning history for archeological research
- ✅ Reduces noise when browsing the project

### CLI Help

Run `bd <command> --help` to see all available flags for any command.
For example: `bd create --help` shows `--parent`, `--deps`, `--assignee`, etc.

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ✅ Store AI planning docs in `history/` directory
- ✅ Run `bd <cmd> --help` to discover available flags
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems
- ❌ Do NOT clutter repo root with planning documents

### Beads Usage Status

- Repository rules are mirrored in `.beads/instructions.md` for beads tracking. Update this file
  first and copy changes into `.beads/instructions.md` to keep the two instruction sets
  synchronized.
- Beads installer is currently blocked by network restrictions (HTTP 403 from GitHub). Continue
  logging installer status and repository checkpoints in `.beads/database.md` so the beads
  database can be seeded once connectivity is restored.
