"""Parse public structured reasoning metadata from model outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping

from .schema import ReasoningGraph

SECTION_NAMES = ("brainstorm", "graph", "graph_json", "patterns", "synthesis")


@dataclass
class PhaseParseResult:
    """Parsed public reasoning sections plus diagnostics."""

    sections: Dict[str, str] = field(default_factory=dict)
    graph: ReasoningGraph | None = None
    diagnostics: List[str] = field(default_factory=list)


def parse_public_reasoning_phases(output: str) -> PhaseParseResult:
    """Extract public tags without assuming access to hidden chain-of-thought."""

    sections: Dict[str, str] = {}
    diagnostics: List[str] = []
    for name in SECTION_NAMES:
        match = re.search(rf"<{name}>(.*?)</{name}>", output, flags=re.DOTALL)
        if match:
            sections[name] = match.group(1).strip()
    graph = None
    if "graph_json" in sections:
        try:
            parsed = json.loads(sections["graph_json"])
            if not isinstance(parsed, Mapping):
                raise TypeError(f"expected object, got {type(parsed).__name__}")
            graph = ReasoningGraph.from_mapping(parsed)
        except (TypeError, ValueError, AttributeError, json.JSONDecodeError) as exc:
            diagnostics.append(f"invalid graph_json: {exc}")
    return PhaseParseResult(sections=sections, graph=graph, diagnostics=diagnostics)


__all__ = ["PhaseParseResult", "parse_public_reasoning_phases"]
