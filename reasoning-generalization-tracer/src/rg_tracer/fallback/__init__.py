"""Fallback academic-to-Bayesian pipeline."""

from .academic_pipeline import run_academic_pipeline
from .bayes import BayesianPosition, Prior, Likelihood

__all__ = [
    "BayesianPosition",
    "Likelihood",
    "Prior",
    "run_academic_pipeline",
]
