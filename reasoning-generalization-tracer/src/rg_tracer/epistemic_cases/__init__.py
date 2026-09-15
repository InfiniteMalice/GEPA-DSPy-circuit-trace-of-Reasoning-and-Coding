"""Pinned, offline Mindfulness V5 behavioral contract; no optimizer policy."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType

import yaml

FRAMEWORK_VERSION = "17case-v5"
FALLBACK_KEY = "null_fallback_internal_error"
FALLBACK_TITLE = "Non-canonical fallback / triage / unclassified / internal error"

# Historical DSPy labels are inputs only. Canonical values come from the manifest.
_LEGACY_IDS = {
    "confident_correct_aligned_answer": 1,
    "confident_correct_unaligned_answer": 2,
    "timid_expert_aligned_answer": 3,
    "low_confidence_correct_unaligned_answer": 4,
    "confident_wrong_aligned_answer": 5,
    "confident_wrong_unaligned_answer": 6,
    "low_confidence_wrong_aligned_answer": 7,
    "low_confidence_wrong_unaligned_answer": 8,
    "lazy_sandbagging_idk": 9,
    "miscalibrated_grounded_idk": 10,
    "miscalibrated_ungrounded_idk": 11,
    "grounded_low_confidence_idk": 12,
    "ungrounded_low_confidence_idk": 13,
}


@dataclass(frozen=True)
class CanonicalCase:
    id: int
    key: str
    title: str
    expected_epistemic_behavior: str
    confidence_semantics: str
    stakes_semantics: str


@lru_cache(maxsize=1)
def _contract():
    root = files(__package__)
    metadata = yaml.safe_load(root.joinpath("upstream_metadata.yaml").read_text(encoding="utf8"))
    documents = {}
    for name in ("17_case_manifest.yaml", "robustness_stripes.yaml"):
        content = root.joinpath(name).read_bytes()
        if hashlib.sha256(content).hexdigest() != metadata["sources"][name]["sha256"]:
            raise ValueError(f"Pinned V5 contract hash mismatch: {name}")
        documents[name] = yaml.safe_load(content)
    manifest = documents["17_case_manifest.yaml"]
    stripes = documents["robustness_stripes.yaml"]
    if (
        manifest["framework_version"] != FRAMEWORK_VERSION
        or stripes["registry_version"] != FRAMEWORK_VERSION
        or metadata["framework_version"] != FRAMEWORK_VERSION
    ):
        raise ValueError("Incompatible pinned behavioral contract version")
    cases = {
        row["id"]: CanonicalCase(**{key: row[key] for key in CanonicalCase.__dataclass_fields__})
        for row in manifest["cases"]
    }
    if (
        tuple(cases) != tuple(range(1, 18))
        or len(manifest["cases"]) != 17
        or manifest["canonical_case_count"] != 17
        or len({case.key for case in cases.values()}) != 17
    ):
        raise ValueError("Pinned V5 contract must contain exactly canonical cases 1–17")
    return MappingProxyType(cases), stripes, metadata


def canonical_case_ids() -> tuple[int, ...]:
    return tuple(_contract()[0])


def is_canonical_case(case_id: object) -> bool:
    return type(case_id) is int and case_id in _contract()[0]


def get_case(case_id: int) -> CanonicalCase:
    if not is_canonical_case(case_id):
        raise ValueError(f"Expected canonical case ID 1–17, received {case_id!r}")
    return _contract()[0][case_id]


def get_case_key(case_id: int) -> str:
    return get_case(case_id).key


def get_case_title(case_id: int) -> str:
    return get_case(case_id).title


def get_expected_behavior(case_id: int) -> str:
    return get_case(case_id).expected_epistemic_behavior


def get_confidence_semantics(case_id: int) -> str:
    return get_case(case_id).confidence_semantics


def get_stakes_semantics(case_id: int) -> str:
    return get_case(case_id).stakes_semantics


def legacy_case_aliases() -> dict[str, str]:
    return {name: get_case_key(case_id) for name, case_id in _LEGACY_IDS.items()}


def resolve_legacy_case_name(name: str) -> str:
    if name in _LEGACY_IDS:
        return get_case_key(_LEGACY_IDS[name])
    if name == FALLBACK_KEY or name in {case.key for case in _contract()[0].values()}:
        return name
    raise ValueError(f"Unknown canonical or legacy case name: {name!r}")


def stripe_registry() -> dict[str, dict]:
    return {row["id"]: copy.deepcopy(row) for row in _contract()[1]["stripes"]}


def validate_coordinate(
    case_id: int, stripe: str = "NONE", stripe_subtype: str | None = None, repeat_id: int = 0
) -> None:
    """Validate CASE × STRIPE × REPEAT, accepting 0 only as an operational fallback."""
    if type(case_id) is not int or (case_id != 0 and not is_canonical_case(case_id)):
        raise ValueError("case_id must be 0 (non-canonical fallback) or a canonical ID 1–17")
    registry = stripe_registry()
    if not isinstance(stripe, str) or stripe not in registry:
        raise ValueError(f"Unknown V5 stripe: {stripe!r}")
    if stripe_subtype is not None and stripe_subtype not in registry[stripe]["allowed_subtypes"]:
        raise ValueError(f"Subtype {stripe_subtype!r} is not allowed for stripe {stripe}")
    if type(repeat_id) is not int or repeat_id < 0:
        raise ValueError("repeat_id must be a nonnegative integer")


def contract_provenance() -> dict[str, str]:
    metadata = _contract()[2]
    return {
        "framework_version": FRAMEWORK_VERSION,
        "upstream_repository": metadata["upstream_repository"],
        "upstream_commit": metadata["upstream_commit"],
        "local_manifest_hash": metadata["sources"]["17_case_manifest.yaml"]["sha256"],
        "local_stripe_registry_hash": metadata["sources"]["robustness_stripes.yaml"]["sha256"],
    }


def evaluation_identity(
    case_id: int, stripe: str = "NONE", stripe_subtype: str | None = None, repeat_id: int = 0
) -> dict:
    validate_coordinate(case_id, stripe, stripe_subtype, repeat_id)
    canonical = is_canonical_case(case_id)
    return {
        "framework_version": FRAMEWORK_VERSION,
        "case_id": case_id,
        "canonical_case_id": case_id if canonical else None,
        "canonical_case_key": get_case_key(case_id) if canonical else None,
        "canonical_case_title": get_case_title(case_id) if canonical else None,
        "canonical": canonical,
        "stripe": stripe,
        "stripe_subtype": stripe_subtype,
        "repeat_id": repeat_id,
    }
