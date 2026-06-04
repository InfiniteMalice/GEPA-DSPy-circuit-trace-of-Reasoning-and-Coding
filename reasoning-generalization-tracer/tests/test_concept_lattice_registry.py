"""Tests for shadow-only concept lattice registry behavior."""

from __future__ import annotations

import pytest

from rg_tracer.concept_lattices import ConceptLatticeRegistry
from rg_tracer.concept_lattices.examples import EXAMPLE_CASES, synthetic_examples


def test_registry_rejects_duplicate_lattice_names():
    registry = ConceptLatticeRegistry()
    spec = synthetic_examples()[0]
    registry.register(spec)
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(spec)


def test_synthetic_examples_reference_existing_attributes():
    specs = {spec.name: spec for spec in synthetic_examples()}
    for example in EXAMPLE_CASES:
        attributes = {attribute.name for attribute in specs[example.lattice_name].attributes}
        assert set(example.expected_attributes) <= attributes
