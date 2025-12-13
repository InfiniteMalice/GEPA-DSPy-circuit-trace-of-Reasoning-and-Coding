"""Bayesian utilities for the fallback pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Prior:
    hypothesis: str
    probability: float


@dataclass
class Likelihood:
    evidence: str
    probability_if_true: float
    probability_if_false: float


@dataclass
class BayesianPosition:
    prior: Prior
    likelihoods: List[Likelihood]
    posterior: float
    sensitivity_summary: str
    dominant_evidence: str
    decision_policy: str


def compute_posterior(prior: Prior, likelihoods: Iterable[Likelihood]) -> BayesianPosition:
    likes = list(likelihoods)
    prob_true = prior.probability
    prob_false = 1 - prob_true
    logit = prob_true / max(prob_false, 1e-9)
    dominant = None
    for like in likes:
        if like.probability_if_false <= 0 or like.probability_if_true <= 0:
            continue
        ratio = like.probability_if_true / like.probability_if_false
        logit *= ratio
        if dominant is None or ratio > dominant[0]:
            dominant = (ratio, like.evidence)
    posterior = logit / (1 + logit)
    posterior = min(max(posterior, 0.0), 1.0)
    sensitivity = "Posterior sensitive to dominant likelihood." if dominant else "Stable"
    dominant_desc = dominant[1] if dominant else "None"
    policy = "Gather more evidence"
    if posterior >= 0.75:
        policy = "Support with caveats"
    elif posterior <= 0.25:
        policy = "Recommend caution"
    return BayesianPosition(
        prior=prior,
        likelihoods=likes,
        posterior=posterior,
        sensitivity_summary=sensitivity,
        dominant_evidence=dominant_desc,
        decision_policy=policy,
    )


__all__ = [
    "Prior",
    "Likelihood",
    "BayesianPosition",
    "compute_posterior",
]
