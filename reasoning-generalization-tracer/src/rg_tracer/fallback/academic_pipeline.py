"""Three-step academic pipeline culminating in a Bayesian position."""

from __future__ import annotations

import json
from collections.abc import Iterable as IterableABC
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Sequence

from ..abstention import ABSTENTION_THRESHOLD
from ..humanities import HumanitiesProfile, evaluate_profile, load_profiles
from ..semantics import SemanticTag, verify_chain
from .bayes import Likelihood, Prior, BayesianPosition, compute_posterior


def _load_records(path: str | Path) -> List[Mapping[str, object]]:
    records: List[Mapping[str, object]] = []
    with Path(path).open("r", encoding="utf8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _iter_tag_labels(entry: Mapping[str, object]) -> Iterable[str]:
    """Yield clean tag labels from ``entry``.

    ``entry["tags"]`` may be a single string or any iterable of strings. Other
    payload shapes are treated as empty. Non-string elements are filtered out to
    guarantee downstream consumers always receive string labels.
    """
    raw_tags = entry.get("tags", [])
    if isinstance(raw_tags, str):
        values: Iterable[object] = [raw_tags]
    elif isinstance(raw_tags, IterableABC):
        values = raw_tags
    else:
        values = []
    for label in values:
        if isinstance(label, str):
            yield label


def _count_tag(report_tags: Sequence[Mapping[str, object]], tag: SemanticTag) -> int:
    def _entry_has_tag(entry: Mapping[str, object]) -> bool:
        return any(label == tag.value for label in _iter_tag_labels(entry))

    return sum(1 for entry in report_tags if _entry_has_tag(entry))


def _grade_ratio(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    if value >= 0.8:
        return 0.98
    if value >= 0.5:
        return 0.9
    if value > 0.0:
        return 0.75
    return 0.2


def _build_metrics(report: Mapping[str, object]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    fallacy_flags = float(report.get("fallacy_flags", 0))
    neutral = float(report.get("neutrality_balance", 0.0))
    citation = float(report.get("citation_coverage", 0.0))
    quote_presence_value = report.get("quote_presence")
    if quote_presence_value is None:
        quote_presence_value = report.get("quote_integrity")
    if quote_presence_value is None:
        quote_presence_value = report.get("quotes")
    if isinstance(quote_presence_value, (int, float, bool, str)):
        try:
            quote_presence = float(quote_presence_value)
        except ValueError:
            quote_presence = 0.0
    else:
        quote_presence = 0.0
    counter = float(report.get("counterevidence_ratio", 0.0))
    hedge = float(report.get("hedge_rate", 0.0))
    fact_free = float(report.get("fact_free_ratio", 0.0))
    contradiction = float(report.get("contradiction_rate", 0.0))
    entailed = float(report.get("entailed_steps_pct", 0.0))
    schema = float(report.get("schema_consistency_pct", 0.0))
    fallacy_ratio = 1.0 / (1.0 + fallacy_flags)
    combined_sources = max(citation, quote_presence)
    metrics["source_handling"] = {
        "positive": _grade_ratio(citation),
        "coverage": _grade_ratio(combined_sources),
        "penalties": float(report.get("misquote_penalty", 0)),
    }
    metrics["interpretive_fidelity"] = {
        "positive": _grade_ratio(1 - fact_free),
        "coverage": _grade_ratio(counter),
        "penalties": float(report.get("quote_ooc_penalty", 0)),
    }
    metrics["historiography_context"] = {
        "positive": _grade_ratio(schema),
        "coverage": _grade_ratio(neutral),
    }
    metrics["causal_discipline"] = {
        "positive": _grade_ratio(1 - contradiction),
        "coverage": _grade_ratio(1 - fact_free),
        "penalties": float(report.get("causality_penalty", 0)),
    }
    metrics["triangulation"] = {
        "positive": _grade_ratio(citation),
        "coverage": _grade_ratio(counter),
    }
    metrics["normative_positive_sep"] = {
        "positive": _grade_ratio(hedge),
        "coverage": _grade_ratio(1 - fact_free),
        "penalties": float(report.get("is_ought_penalty", 0)),
    }
    metrics["uncertainty_calibration"] = {
        "positive": _grade_ratio(hedge),
        "coverage": _grade_ratio(1 - contradiction),
    }
    metrics["intellectual_charity"] = {
        "positive": _grade_ratio(neutral),
        "coverage": _grade_ratio(counter),
    }
    metrics["rhetorical_hygiene"] = {
        "positive": _grade_ratio(fallacy_ratio),
        "coverage": _grade_ratio(1 - contradiction),
        "penalties": float(report.get("fallacy_flags", 0)),
    }
    metrics["reproducibility_transparency"] = {
        "positive": _grade_ratio(citation),
        "coverage": _grade_ratio(combined_sources),
    }
    metrics["synthesis_generalization"] = {
        "positive": _grade_ratio(schema),
        "coverage": _grade_ratio(entailed),
    }
    metrics["epistemic_neutrality"] = {
        "positive": _grade_ratio(neutral),
        "coverage": _grade_ratio(1 - contradiction),
    }
    return metrics


def _evidence_table(record: Mapping[str, object]) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for item in record.get("evidence", []):
        entries.append(
            {
                "source": item.get("source", "unknown"),
                "year": item.get("year", ""),
                "method": item.get("method", ""),
                "finding": item.get("finding", ""),
                "limitations": item.get("limitations", ""),
            }
        )
    return entries


def _likelihoods(record: Mapping[str, object]) -> List[Likelihood]:
    likes: List[Likelihood] = []
    for item in record.get("evidence", []):
        likes.append(
            Likelihood(
                evidence=item.get("finding", ""),
                probability_if_true=float(item.get("support_if_true", 0.7)),
                probability_if_false=float(item.get("support_if_false", 0.3)),
            )
        )
    return likes


def run_academic_pipeline(
    problem_path: str | Path,
    *,
    profile: str = "humanities",
) -> Dict[str, object]:
    records = _load_records(problem_path)
    if not records:
        raise ValueError("No problems found for fallback pipeline")
    profiles = load_profiles()
    if profile not in profiles:
        raise KeyError(f"Humanities profile {profile} not found")
    profile_obj: HumanitiesProfile = profiles[profile]

    processed: List[Dict[str, object]] = []
    posteriors: List[float] = []
    abstentions = 0

    for record in records:
        steps = record.get("analysis") or [record.get("claim", "")]
        steps = [str(step) for step in steps if str(step).strip()]
        chain = "\n".join(steps)
        report = verify_chain(
            chain,
            {
                "concept": record.get("concept", ""),
                "domain": "humanities",
            },
        )
        tags = report.tags
        penalties = {
            "misquote_penalty": _count_tag(tags, SemanticTag.MISQUOTE),
            "quote_ooc_penalty": _count_tag(tags, SemanticTag.QUOTE_OOC),
            "causality_penalty": _count_tag(tags, SemanticTag.OVERCLAIMED_CAUSALITY),
            "is_ought_penalty": _count_tag(tags, SemanticTag.IS_OUGHT_SLIP),
        }
        metrics = {**report.as_dict(), **penalties}
        humanities_metrics = _build_metrics(metrics)
        evaluation = evaluate_profile(humanities_metrics, profile_obj)
        composite = float(evaluation["composite"])
        confidence = min(0.99, max(0.0, composite / 1.5))
        passes = bool(evaluation["passes_gates"])
        abstained = not passes or confidence < ABSTENTION_THRESHOLD
        record_result: Dict[str, object] = {
            "id": record.get("id"),
            "analysis": steps,
            "humanities": evaluation,
            "abstained": abstained,
            "confidence": confidence,
            "evidence_table": _evidence_table(record),
        }
        if abstained:
            abstentions += 1
            reasons = []
            if not passes:
                reasons.append("Humanities hard gates failed")
            if confidence < ABSTENTION_THRESHOLD:
                reasons.append("Confidence below calibrated threshold")
            record_result["decision"] = "I don't know."
            record_result["reasons"] = reasons
        else:
            prior_value = float(record.get("prior", 0.5))
            prior = Prior(hypothesis=record.get("claim", ""), probability=prior_value)
            likes = _likelihoods(record)
            position: BayesianPosition = compute_posterior(prior, likes)
            record_result["posterior"] = position.posterior
            record_result["decision_policy"] = position.decision_policy
            record_result["bayesian"] = {
                "sensitivity_summary": position.sensitivity_summary,
                "dominant_evidence": position.dominant_evidence,
            }
            posteriors.append(position.posterior)
        processed.append(record_result)

    summary = {
        "count": len(processed),
        "abstentions": abstentions,
        "mean_posterior": mean(posteriors) if posteriors else 0.0,
    }
    return {"records": processed, "summary": summary}


__all__ = ["run_academic_pipeline"]
