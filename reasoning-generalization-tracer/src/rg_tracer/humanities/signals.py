"""Signal extraction for humanities-style reasoning chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, TypedDict

from ..semantics import SemanticTag


class StepTagMapping(TypedDict):
    step: str
    tags: List[str]


@dataclass
class HumanitiesSignals:
    citation_coverage: float
    quote_presence: float
    counterevidence_ratio: float
    hedge_rate: float
    fallacy_flags: int
    neutrality_balance: float
    # Each tag mapping exposes the original step alongside a list of tag labels.
    tags: List[StepTagMapping]

    def as_dict(self) -> Mapping[str, object]:
        return {
            "citation_coverage": self.citation_coverage,
            "quote_presence": self.quote_presence,
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
_ASSERTIVE_TERMS = {
    "shows",
    "demonstrates",
    "proves",
    "confirms",
    "reveals",
    "therefore",
}


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
    tags: List[StepTagMapping] = []
    for step in steps:
        lowered = step.lower()
        step_tags: List[str] = []
        has_citation = any(marker in step for marker in _CITATION_MARKERS)
        if has_citation:
            cite_hits += 1
        quote_count = step.count('"')
        if quote_count:
            quote_hits += 1
        if any(term in lowered for term in _COUNTER_TERMS):
            counter_hits += 1
        if any(term in lowered for term in _HEDGE_TERMS):
            hedge_hits += 1
        if any(term in lowered for term in _FALLACY_TERMS):
            fallacy_flags += 1
            step_tags.append(SemanticTag.RHETORICAL_EXCESS.value)
        if not has_citation and any(term in lowered for term in _ASSERTIVE_TERMS):
            step_tags.append(SemanticTag.UNCITED_CLAIM.value)
        quote_issue = False
        if quote_count % 2 == 1:
            quote_issue = True
        if quote_issue:
            step_tags.append(SemanticTag.MISQUOTE.value)
        if ('"' in step) and not has_citation:
            step_tags.append(SemanticTag.QUOTE_OOC.value)
        if "balance" in lowered or "both" in lowered:
            neutrality_hits += 1
        tags.append({"step": step, "tags": step_tags})
    if counter_hits == 0 and steps and tags:
        unsupported = SemanticTag.UNSUPPORTED.value
        existing = list(tags[-1]["tags"])
        if unsupported not in existing:
            existing.append(unsupported)
            tags[-1]["tags"] = existing
    total = len(steps)
    return HumanitiesSignals(
        citation_coverage=cite_hits / total,
        quote_presence=quote_hits / total,
        counterevidence_ratio=counter_hits / total,
        hedge_rate=hedge_hits / total,
        fallacy_flags=fallacy_flags,
        neutrality_balance=neutrality_hits / total,
        tags=tags,
    )


__all__ = ["HumanitiesSignals", "StepTagMapping", "analyse_humanities_chain"]
