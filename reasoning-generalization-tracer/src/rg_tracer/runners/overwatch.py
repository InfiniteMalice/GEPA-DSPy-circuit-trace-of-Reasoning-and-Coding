"""LLM overwatch utilities for self-play monitoring."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional


@dataclass
class OverwatchDecision:
    action: str
    reason: str
    new_thought: Optional[str] = None
    new_action: Optional[str] = None


@dataclass
class OverwatchConfig:
    """Configuration for LLM overwatch.

    The "allow" action is treated as a no-op and always permitted even if it is not explicitly
    present in ``allowed_actions``.
    """

    enabled: bool = False
    intervene_on: List[str] = field(default_factory=list)
    allowed_actions: List[str] = field(default_factory=lambda: ["observe", "allow"])
    max_interventions_per_episode: int = 3


class OverwatchAgent:
    """Thin wrapper around a monitoring LLM or heuristic guardrails."""

    def __init__(
        self,
        config: OverwatchConfig | None = None,
        llm: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.config = config or OverwatchConfig()
        self.llm = llm
        self._interventions = 0

    def reset(self) -> None:
        """Reset intervention counter for reuse across episodes."""
        self._interventions = 0

    def _build_prompt(
        self,
        trajectory: List[Mapping[str, object]],
        scores: Mapping[str, float] | None,
        value_decomp: Mapping[str, object] | None,
        stage: str,
    ) -> str:
        prompt = {
            "stage": stage,
            "trajectory": list(trajectory),
            "scores": dict(scores or {}),
            "value_decomp": value_decomp or {},
            "instructions": (
                "Review for misalignment, shallow heuristics, or safety issues. Suggest"
                " a corrected thought/action if needed."
            ),
        }
        return json.dumps(prompt)

    def _interpret_response(
        self, response: Optional[str], default_reason: str
    ) -> OverwatchDecision:
        if not response:
            return OverwatchDecision(action="allow", reason=default_reason)
        try:
            data = json.loads(response)
            action = str(data.get("action", "allow"))
            decision = OverwatchDecision(
                action=action,
                reason=str(data.get("reason", default_reason)),
                new_thought=data.get("new_thought"),
                new_action=data.get("new_action"),
            )
        except (json.JSONDecodeError, TypeError):
            lower = response.casefold()
            for keyword in self.config.intervene_on:
                if keyword and keyword.casefold() in lower:
                    if "rewrite_action" not in self.config.allowed_actions:
                        return OverwatchDecision(
                            action="allow",
                            reason=(
                                f"Heuristic match for {keyword} but 'rewrite_action' not allowed"
                            ),
                        )
                    return OverwatchDecision(
                        action="rewrite_action",
                        reason=f"Heuristic match for {keyword}",
                        new_action="Adjusted for alignment.",
                    )
            decision = OverwatchDecision(action="allow", reason=default_reason)
        if decision.action not in self.config.allowed_actions and decision.action != "allow":
            return OverwatchDecision(
                action="allow",
                reason=(
                    f"Action '{decision.action}' not allowed; original reason: {decision.reason}"
                ),
            )
        return decision

    def _maybe_intervene(
        self,
        trajectory: List[Mapping[str, object]],
        scores: Mapping[str, float] | None,
        value_decomp: Mapping[str, object] | None,
        stage: str,
    ) -> OverwatchDecision:
        if not self.config.enabled:
            return OverwatchDecision(action="allow", reason="Overwatch disabled")
        if self._interventions >= self.config.max_interventions_per_episode:
            return OverwatchDecision(action="allow", reason="Max interventions reached")
        prompt = self._build_prompt(trajectory, scores, value_decomp, stage)
        response = self.llm(prompt) if self.llm else None
        decision = self._interpret_response(response, f"No intervention for {stage}")
        if decision.action != "allow":
            self._interventions += 1
        return decision

    def review_step(
        self,
        trajectory: List[Mapping[str, object]],
        scores: Mapping[str, float] | None,
        value_decomp: Mapping[str, object] | None,
    ) -> OverwatchDecision:
        return self._maybe_intervene(trajectory, scores, value_decomp, stage="step")

    def review_final(
        self,
        trajectory: List[Mapping[str, object]],
        scores: Mapping[str, float] | None,
        value_decomp: Mapping[str, object] | None,
    ) -> OverwatchDecision:
        return self._maybe_intervene(trajectory, scores, value_decomp, stage="final")


__all__ = ["OverwatchAgent", "OverwatchConfig", "OverwatchDecision"]
