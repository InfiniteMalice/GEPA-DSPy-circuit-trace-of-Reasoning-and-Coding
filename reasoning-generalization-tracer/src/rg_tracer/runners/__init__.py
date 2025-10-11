"""Runner utilities for RG-Tracer."""

from .self_play import run_self_play, pareto_frontier
from .eval_suite import evaluate_dataset

__all__ = ["run_self_play", "pareto_frontier", "evaluate_dataset"]
