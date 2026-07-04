"""Humanities-specific scoring and signals for neutral-but-rigorous analysis."""

from .axes import HUMANITIES_AXES, HumanitiesScores
from .aggregator import HumanitiesProfile, load_profiles, evaluate_profile
from .signals import analyse_humanities_chain

__all__ = [
    "HUMANITIES_AXES",
    "HumanitiesProfile",
    "HumanitiesScores",
    "analyse_humanities_chain",
    "evaluate_profile",
    "load_profiles",
]
