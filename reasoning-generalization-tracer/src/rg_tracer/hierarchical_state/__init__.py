"""Hierarchical planning and information-folding metadata."""

from .folding import fold_subgoals
from .state_summary import FoldedState
from .subgoals import Subgoal

__all__ = ["FoldedState", "Subgoal", "fold_subgoals"]
