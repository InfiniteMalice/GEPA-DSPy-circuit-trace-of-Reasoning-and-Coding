"""Case counts with explicit fallback totals and independent evaluation slices."""

from collections import Counter
from collections.abc import Iterable, Mapping

from . import FRAMEWORK_VERSION, canonical_case_ids, validate_coordinate


def summarize_cases(records: Iterable[Mapping]) -> dict:
    """Read current case IDs or legacy reward_case fields without relabeling provenance."""
    counts = dict.fromkeys(canonical_case_ids(), 0)
    slices = Counter()
    fallback = 0
    for record in records:
        if record.get("framework_version", FRAMEWORK_VERSION) != FRAMEWORK_VERSION:
            raise ValueError("Cannot count another framework version as canonical V5")
        case_id = record.get("case_id", record.get("canonical_case_id", record.get("reward_case")))
        if case_id is None:
            case_id = 0
        stripe = record.get("stripe", "NONE")
        subtype = record.get("stripe_subtype")
        repeat = record.get("repeat_id", 0)
        validate_coordinate(case_id, stripe, subtype, repeat)
        expected_canonical_id = case_id if case_id else None
        if "canonical_case_id" in record:
            value = record["canonical_case_id"]
            if type(value) is not type(expected_canonical_id) or value != expected_canonical_id:
                raise ValueError("canonical_case_id contradicts case_id")
        if "canonical" in record and record["canonical"] is not (case_id != 0):
            raise ValueError("canonical flag contradicts case_id")
        if case_id == 0:
            fallback += 1
            continue
        counts[case_id] += 1
        slices[(case_id, stripe, subtype, repeat)] += 1
    return {
        "case_counts": counts,
        "canonical_count": sum(counts.values()),
        "unclassified_count": fallback,
        "coordinate_counts": [
            {
                "canonical_case_id": key[0],
                "stripe": key[1],
                "stripe_subtype": key[2],
                "repeat_id": key[3],
                "count": count,
            }
            for key, count in sorted(slices.items(), key=lambda item: repr(item[0]))
        ],
    }
