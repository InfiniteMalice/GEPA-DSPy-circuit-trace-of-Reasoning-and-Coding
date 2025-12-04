# GEPA-DSPy Circuit Tracing Rules

These rules apply to the entire repository. All code changes must follow them.

## Critical Rules
1. **Python files use a 100 character line limit.** No exceptions.
2. **Import organization** must follow the pattern:
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
3. **No unused imports.** Verify every DSPy module imported is used.
4. **Black formatting** with `black --line-length 100 .`.

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

## Beads Usage
- Repository rules are mirrored in `.beads/instructions.md` for beads tracking.
- Update this file first and copy changes into `.beads/instructions.md` to keep the two
  instruction sets synchronized.
- Install beads with
  `curl -fsSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash`
  when network access allows.
- If installation fails because GitHub is blocked, retry when access is restored.
