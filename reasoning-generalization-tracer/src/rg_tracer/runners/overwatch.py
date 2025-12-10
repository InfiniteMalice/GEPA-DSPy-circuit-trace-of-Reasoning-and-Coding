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
        except json.JSONDecodeError:
            data = None
        if data is not None and isinstance(data, Mapping):
            action = str(data.get("action", "allow"))
            decision = OverwatchDecision(
                action=action,
                reason=str(data.get("reason", default_reason)),
                new_thought=str(data["new_thought"]) if data.get("new_thought") else None,
                new_action=str(data["new_action"]) if data.get("new_action") else None,
            )
        else:
            decision = self._apply_heuristic_fallback(response, default_reason)
        if decision.action not in self.config.allowed_actions and decision.action != "allow":
            return OverwatchDecision(
                action="allow",
                reason=(
                    f"Action '{decision.action}' not allowed; original reason: {decision.reason}"
                ),
            )
        return decision

    def _apply_heuristic_fallback(
        self, response: str, default_reason: str
    ) -> OverwatchDecision:
        """Apply heuristic keyword matching when LLM returns non-JSON."""

        lower = response.casefold()
        for keyword in self.config.intervene_on:
            if keyword and keyword.casefold() in lower:
                allowed_interventions = [
                    action
                    for action in self.config.allowed_actions
                    if action not in {"allow", "observe"}
                ]
                if not allowed_interventions:
                    return OverwatchDecision(
                        action="allow",
                        reason=(
                            "Heuristic match but no intervention actions are permitted; "
                            f"keyword={keyword}"
                        ),
                    )
                action = allowed_interventions[0]
                if action == "rewrite_thought":
                    return OverwatchDecision(
                        action="rewrite_thought",
                        reason=f"Heuristic match for {keyword}",
                        new_thought="Aligned thought for safety.",
                    )
                if action == "abort_episode":
                    return OverwatchDecision(
                        action="abort_episode",
                        reason=f"Heuristic match for {keyword}",
                    )
                if action == "rewrite_action":
                    return OverwatchDecision(
                        action="rewrite_action",
                        reason=f"Heuristic match for {keyword}",
                        new_action="Adjusted for alignment.",
                    )
                return OverwatchDecision(
                    action="allow",
                    reason=(
                        "Heuristic match but first allowed action is unrecognized: "
                        f"{action}"
                    ),
                )
        return OverwatchDecision(action="allow", reason=default_reason)

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
        if decision.action not in {"allow", "observe"}:
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
