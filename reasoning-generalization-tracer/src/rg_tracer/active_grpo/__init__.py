"""Optional Active-GRPO reference updating helpers."""

from .reference_store import JSONLReferenceStore
from .strategy import decide_active_grpo, maybe_update_reference
from .types import ActiveGRPODecision, CandidateRecord, ReferenceRecord

__all__ = [
    "ActiveGRPODecision",
    "CandidateRecord",
    "JSONLReferenceStore",
    "ReferenceRecord",
    "decide_active_grpo",
    "maybe_update_reference",
]
