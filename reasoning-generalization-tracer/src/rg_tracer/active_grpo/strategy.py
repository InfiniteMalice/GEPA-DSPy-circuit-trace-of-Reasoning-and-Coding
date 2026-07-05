"""Active-GRPO-inspired reference management strategy."""

from __future__ import annotations

from .reference_store import JSONLReferenceStore
from .types import ActiveGRPODecision, CandidateRecord, ReferenceRecord, utc_now


def decide_active_grpo(
    reference: ReferenceRecord | None,
    candidate: CandidateRecord,
    margin: float = 0.0,
    require_verified: bool = True,
) -> ActiveGRPODecision:
    """Decide whether to imitate a reference or reinforce a verified candidate."""

    if reference is None:
        return ActiveGRPODecision(
            mode="no_reference",
            reason="no reference exists for this task",
            score_delta=candidate.candidate_score,
            should_update_reference=bool(candidate.verified or not require_verified),
        )
    delta = candidate.candidate_score - reference.reference_score
    if require_verified and not candidate.verified:
        return ActiveGRPODecision(
            mode="reject_candidate",
            reason="candidate is not verified",
            score_delta=delta,
            should_update_reference=False,
        )
    if delta > margin:
        return ActiveGRPODecision(
            mode="reinforce_candidate",
            reason="verified candidate exceeds reference by configured margin",
            score_delta=delta,
            should_update_reference=True,
        )
    return ActiveGRPODecision(
        mode="imitate_reference",
        reason="reference remains at least as strong as candidate under margin",
        score_delta=delta,
        should_update_reference=False,
    )


def maybe_update_reference(
    store: JSONLReferenceStore,
    reference: ReferenceRecord | None,
    candidate: CandidateRecord,
    decision: ActiveGRPODecision,
) -> ReferenceRecord | None:
    """Append a new reference record when admission policy chooses an upgrade."""

    if not decision.should_update_reference:
        return None
    existing = reference
    lineage = list(existing.lineage) if existing else []
    if existing:
        lineage.append(existing.reference_answer)
    record = ReferenceRecord(
        task_id=candidate.task_id,
        reference_answer=candidate.candidate_answer,
        reference_score=candidate.candidate_score,
        source="active_grpo_candidate",
        created_at=existing.created_at if existing else utc_now(),
        updated_at=utc_now(),
        lineage=lineage,
        metadata={
            "decision_mode": decision.mode,
            "score_delta": decision.score_delta,
            "candidate_metadata": dict(candidate.metadata),
        },
    )
    store.append(record)
    return record


__all__ = ["decide_active_grpo", "maybe_update_reference"]
