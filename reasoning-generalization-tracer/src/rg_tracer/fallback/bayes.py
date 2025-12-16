"""Bayesian utilities for the fallback pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, isinf, log
from typing import Iterable


PRIOR_PROBABILITY_INVALID = "prior.probability must be a finite value in [0.0, 1.0]"
LIKELIHOOD_PROBABILITY_INVALID = "Likelihood.{} must be finite and in [0.0, 1.0]"
POSTERIOR_NON_FINITE = "Posterior became non-finite; check likelihood inputs."
CONFLICTING_LIKELIHOODS = (
    "Conflicting likelihoods: some evidence forces posterior=0 and others force posterior=1."
)


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
    likelihoods: list[Likelihood]
    posterior: float
    sensitivity_summary: str
    dominant_evidence: str
    decision_policy: str


def compute_posterior(prior: Prior, likelihoods: Iterable[Likelihood]) -> BayesianPosition:
    if not isfinite(prior.probability) or not 0.0 <= prior.probability <= 1.0:
        raise ValueError(PRIOR_PROBABILITY_INVALID)
    likes = list(likelihoods)
    prob_true = prior.probability
    prob_false = 1 - prob_true
    if prob_true == 0.0:
        log_odds = float("-inf")
    elif prob_true == 1.0:
        log_odds = float("inf")
    else:
        log_odds = log(prob_true) - log(prob_false)
    prior_pins_posterior = isinf(log_odds)

    forced_zero = False
    forced_one = False
    dominant: tuple[float, str] | None = None
    for like in likes:
        for value, name in (
            (like.probability_if_true, "probability_if_true"),
            (like.probability_if_false, "probability_if_false"),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(LIKELIHOOD_PROBABILITY_INVALID.format(name))
        if like.probability_if_true == 0.0 and like.probability_if_false > 0.0:
            forced_zero = True
            impact = float("inf")
        elif like.probability_if_false == 0.0 and like.probability_if_true > 0.0:
            forced_one = True
            impact = float("inf")
        elif like.probability_if_true > 0.0 and like.probability_if_false > 0.0:
            log_ratio = log(like.probability_if_true) - log(like.probability_if_false)
            if not prior_pins_posterior:
                log_odds += log_ratio
            impact = abs(log_ratio)
        else:
            # Both probabilities are 0: evidence impossible under either hypothesis; skip.
            continue

        if dominant is None or impact > dominant[0]:
            dominant = (impact, like.evidence)

    if forced_zero and forced_one:
        raise ValueError(CONFLICTING_LIKELIHOODS)
    if not prior_pins_posterior:
        if forced_one:
            log_odds = float("inf")
        elif forced_zero:
            log_odds = float("-inf")

    if log_odds == float("inf"):
        posterior = 1.0
    elif log_odds == float("-inf"):
        posterior = 0.0
    else:
        posterior = (
            1.0 / (1.0 + exp(-log_odds))
            if log_odds >= 0
            else (exp(log_odds) / (1.0 + exp(log_odds)))
        )
    if not isfinite(posterior):
        raise ValueError(POSTERIOR_NON_FINITE)
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
