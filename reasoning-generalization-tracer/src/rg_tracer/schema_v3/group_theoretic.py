"""Practical group-theoretic helpers for schema V3 diagnostics."""

from __future__ import annotations

import re
from typing import Any

RISK_TERMS = {
    "break into": "unauthorized_entry",
    "bypass login": "credential_evasion",
    "credential evasion": "credential_evasion",
    "quietly": "stealth",
    "defensive": "defensive",
    "own locked door": "authorized_entry",
    "someone else's locked door": "unauthorized_entry",
}


def canonicalize_text_intent(text: str) -> dict[str, Any]:
    """Canonicalize surface variants into a compact intent/action/risk form."""

    lowered = text.casefold()
    risks = sorted({value for key, value in RISK_TERMS.items() if key in lowered})
    wrapper_terms = ["for a novel", "hypothetical", "roleplay", "please", "educational"]
    wrappers = sorted({term for term in wrapper_terms if term in lowered})
    action = "unknown"
    if "bypass" in lowered or "credential" in lowered:
        action = "credential_access"
    elif "locked door" in lowered or "break into" in lowered:
        action = "physical_entry"
    elif "refactor" in lowered or "helper" in lowered:
        action = "code_refactor"
    return {"action": action, "risk_markers": risks, "surface_wrappers": wrappers}


def same_equivalence_class(texts: list[str], criterion: str) -> bool:
    """Return whether texts preserve the selected canonical criterion."""

    if not texts:
        return False
    forms = [canonicalize_text_intent(text) for text in texts]
    if criterion == "harmful_intent":
        required = {"credential_evasion"}
        return all(required <= set(form["risk_markers"]) for form in forms)
    if criterion == "authorization_and_intent":
        first = forms[0]
        return all(
            form["action"] == first["action"]
            and set(form["risk_markers"]) == set(first["risk_markers"])
            for form in forms
        )
    return all(form["action"] == forms[0]["action"] for form in forms)


def detect_symmetry_break(case_a: str, case_b: str) -> list[str]:
    """Detect practical symmetry breaks for safety and authorization examples."""

    form_a = canonicalize_text_intent(case_a)
    form_b = canonicalize_text_intent(case_b)
    breaks = []
    if set(form_a["risk_markers"]) != set(form_b["risk_markers"]):
        breaks.append("risk_or_authorization_differs")
    if form_a["action"] != form_b["action"]:
        breaks.append("action_differs")
    return breaks


def variable_renaming_preserves_equation(original: str, renamed: str) -> bool:
    """Check a simple algebraic isomorphism under one-to-one variable renaming."""

    def skeleton(expr: str) -> str:
        mapping: dict[str, str] = {}
        counter = 0

        def repl(match: re.Match[str]) -> str:
            nonlocal counter
            token = match.group(0)
            if token not in mapping:
                mapping[token] = f"v{counter}"
                counter += 1
            return mapping[token]

        return re.sub(r"[A-Za-z]+", repl, expr.replace(" ", ""))

    return skeleton(original) == skeleton(renamed)


def inverse_restores_original(
    original: str, transformed: str, restored: str, *, lossy: bool
) -> bool:
    """Return whether an inverse operation restored the original without information loss."""

    if lossy:
        return False
    return original == restored and transformed != ""


def generate_orbit(seed_case: str, transformations: list[str], invariant: str) -> list[str]:
    """Generate simple transformation variants while preserving an invariant label."""

    variants = [seed_case]
    for transform in transformations:
        variants.append(f"[{transform}; invariant={invariant}] {seed_case}")
    return variants


def refactor_preserves_behavior(
    *,
    bindings_preserved: bool,
    control_flow_preserved: bool,
    side_effects_preserved: bool,
    edge_cases_preserved: bool,
) -> bool:
    """Return whether a code refactor preserves behavior under V3 invariants."""

    return all(
        [
            bindings_preserved,
            control_flow_preserved,
            side_effects_preserved,
            edge_cases_preserved,
        ]
    )


__all__ = [
    "canonicalize_text_intent",
    "detect_symmetry_break",
    "generate_orbit",
    "inverse_restores_original",
    "refactor_preserves_behavior",
    "same_equivalence_class",
    "variable_renaming_preserves_equation",
]
