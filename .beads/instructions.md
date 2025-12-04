# Beads Instructions

These instructions mirror `AGENTS.md` so beads can track repository rules. Update
`AGENTS.md` first and copy changes here to keep both files in sync.

## GEPA-DSPy Circuit Tracing Rules
- Python files use a 100 character line limit. No exceptions.
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

## Beads Setup Notes
- Run beads installer when network access allows:
  `curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash`
- Current attempts fail because GitHub access is blocked by a proxy (HTTP 403).
