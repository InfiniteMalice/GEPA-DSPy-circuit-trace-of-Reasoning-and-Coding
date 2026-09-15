"""Case counts with explicit fallback totals and independent evaluation slices."""

from collections import Counter
from collections.abc import Iterable, Mapping

from . import FRAMEWORK_VERSION, canonical_case_ids, evaluation_identity


def summarize_cases(records: Iterable[Mapping]) -> dict:
    """Count consistent V5 IDs without relabeling supplied identity or provenance.

    Nonnull case_id, canonical_case_id and legacy reward_case must be equal integers.
    Missing/null legacy IDs allow another supplied ID, or Case 0 if none is supplied.
    Supplied canonical fields must match the resolved identity, including null ID,
    key and title for fallback records. Missing canonical fields are accepted.
    """
    counts = dict.fromkeys(canonical_case_ids(), 0)
    slices = Counter()
    fallback = 0
    for record in records:
        if record.get("framework_version", FRAMEWORK_VERSION) != FRAMEWORK_VERSION:
            raise ValueError("Cannot count another framework version as canonical V5")
        supplied_ids = {
            name: record[name]
            for name in ("case_id", "canonical_case_id", "reward_case")
            if record.get(name) is not None
        }
        case_id = next(iter(supplied_ids.values()), 0)
        for name, value in supplied_ids.items():
            if type(value) is not int or value != case_id:
                raise ValueError(f"{name} must be an integer matching the resolved case_id")
        stripe = record.get("stripe", "NONE")
        subtype = record.get("stripe_subtype")
        repeat = record.get("repeat_id", 0)
        identity = evaluation_identity(case_id, stripe, subtype, repeat)
        for name in (
            "canonical_case_id",
            "canonical_case_key",
            "canonical_case_title",
            "canonical",
        ):
            if name in record:
                value = record[name]
                if type(value) is not type(identity[name]) or value != identity[name]:
                    raise ValueError(f"{name} contradicts case_id")
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
