"""Signal extraction for humanities-style reasoning chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping

from ..semantics import SemanticTag


@dataclass
class HumanitiesSignals:
    citation_coverage: float
    quote_integrity: float
    counterevidence_ratio: float
    hedge_rate: float
    fallacy_flags: int
    neutrality_balance: float
    tags: List[Mapping[str, Iterable[str]]]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "citation_coverage": self.citation_coverage,
            "quote_integrity": self.quote_integrity,
            "counterevidence_ratio": self.counterevidence_ratio,
            "hedge_rate": self.hedge_rate,
            "fallacy_flags": self.fallacy_flags,
            "neutrality_balance": self.neutrality_balance,
            "tags": self.tags,
        }


_CITATION_MARKERS = {"[", "("}
_HEDGE_TERMS = {"likely", "perhaps", "suggests", "appears", "contested"}
_COUNTER_TERMS = {"however", "on the other hand", "critics", "counter"}
_FALLACY_TERMS = {"obviously", "clearly", "undeniably", "must", "always"}


def analyse_humanities_chain(chain: Iterable[str]) -> HumanitiesSignals:
    steps = [step.strip() for step in chain if str(step).strip()]
    if not steps:
        steps = [""]
    cite_hits = 0
    quote_hits = 0
    counter_hits = 0
    hedge_hits = 0
    fallacy_flags = 0
    neutrality_hits = 0
    tags: List[Mapping[str, Iterable[str]]] = []
    for step in steps:
        lowered = step.lower()
        if any(marker in step for marker in _CITATION_MARKERS):
            cite_hits += 1
        if '"' in step or "'" in step:
            quote_hits += 1
        if any(term in lowered for term in _COUNTER_TERMS):
            counter_hits += 1
        if any(term in lowered for term in _HEDGE_TERMS):
            hedge_hits += 1
        if any(term in lowered for term in _FALLACY_TERMS):
            fallacy_flags += 1
            tags.append({
                "step": step,
                "tags": [SemanticTag.RHETORICAL_EXCESS.value],
            })
        if "balance" in lowered or "both" in lowered:
            neutrality_hits += 1
    total = len(steps)
    return HumanitiesSignals(
        citation_coverage=cite_hits / total,
        quote_integrity=quote_hits / total,
        counterevidence_ratio=counter_hits / total,
        hedge_rate=hedge_hits / total,
        fallacy_flags=fallacy_flags,
        neutrality_balance=neutrality_hits / total,
        tags=tags,
    )


__all__ = ["HumanitiesSignals", "analyse_humanities_chain"]
