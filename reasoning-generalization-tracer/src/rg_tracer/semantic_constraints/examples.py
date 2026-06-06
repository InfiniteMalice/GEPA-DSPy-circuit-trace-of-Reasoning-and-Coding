"""Small examples for the semantic constraint scaffold."""

from __future__ import annotations


def finite_domain_examples() -> list[dict[str, object]]:
    """Return in-memory examples mirroring the JSONL dataset."""

    return [
        {
            "id": "semantic_even_gt2",
            "task": "semantic_constraint_toy",
            "domain": [1, 2, 3, 4],
            "requirement": "The answer must be even and greater than 2.",
            "answer": 4,
        }
    ]


__all__ = ["finite_domain_examples"]
