"""Shadow-only conceptual lattice scaffolds for future RG-Tracer experiments."""

from .examples import synthetic_examples
from .registry import ConceptLatticeRegistry
from .types import ConceptAttribute, ConceptLatticeExample, ConceptLatticeSpec

__all__ = [
    "ConceptAttribute",
    "ConceptLatticeExample",
    "ConceptLatticeRegistry",
    "ConceptLatticeSpec",
    "synthetic_examples",
]
