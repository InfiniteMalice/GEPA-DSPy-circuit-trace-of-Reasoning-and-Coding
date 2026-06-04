"""Experimental recursive-refinement scaffold for RG-Tracer."""

from .config import LatticeConfig, RecursiveRefinementConfig, RefinementBudget, ViewRoutingConfig
from .controller import RecursiveRefinementController
from .lattice import DeductionConstraint, FiniteCandidateLattice, LatticeElement
from .perturbation import GaussianTrajectoryPerturber, PerturbationConfig
from .projection import LatticeDeductionProjector
from .sampler import GRAMMDTSampler, LatticePTRMSampler, LatticeTRMSampler, PTRMSampler
from .scoring import ProcessScore, score_trajectory
from .types import (
    LatticeDiagnostics,
    PerturbationRecord,
    RefinementRun,
    RefinementState,
    TrajectoryResult,
    ViewOperation,
)
from .views import ReasoningView, build_default_view_registry

__all__ = [
    "DeductionConstraint",
    "FiniteCandidateLattice",
    "GRAMMDTSampler",
    "GaussianTrajectoryPerturber",
    "LatticeConfig",
    "LatticeDeductionProjector",
    "LatticeDiagnostics",
    "LatticeElement",
    "LatticePTRMSampler",
    "LatticeTRMSampler",
    "PTRMSampler",
    "PerturbationConfig",
    "PerturbationRecord",
    "ProcessScore",
    "ReasoningView",
    "RecursiveRefinementConfig",
    "RecursiveRefinementController",
    "RefinementBudget",
    "RefinementRun",
    "RefinementState",
    "TrajectoryResult",
    "ViewOperation",
    "ViewRoutingConfig",
    "build_default_view_registry",
    "score_trajectory",
]
