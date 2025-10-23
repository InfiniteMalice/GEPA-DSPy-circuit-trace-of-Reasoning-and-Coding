"""Fallback academic-to-Bayesian pipeline."""

from .academic_pipeline import run_academic_pipeline
from .bayes import BayesianPosition, Prior, Likelihood

__all__ = [
    "run_academic_pipeline",
    "BayesianPosition",
    "Prior",
    "Likelihood",
]
