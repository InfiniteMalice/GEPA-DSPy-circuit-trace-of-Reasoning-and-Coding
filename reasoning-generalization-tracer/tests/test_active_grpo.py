from rg_tracer.active_grpo import (
    CandidateRecord,
    JSONLReferenceStore,
    ReferenceRecord,
    decide_active_grpo,
    maybe_update_reference,
)


def test_candidate_below_reference_imitates_reference():
    reference = ReferenceRecord("t1", "ref", 0.8, "seed")
    candidate = CandidateRecord("t1", "candidate", 0.7, True)
    decision = decide_active_grpo(reference, candidate)
    assert decision.mode == "imitate_reference"
    assert not decision.should_update_reference


def test_verified_candidate_above_reference_updates_reference(tmp_path):
    store = JSONLReferenceStore(tmp_path / "refs.jsonl")
    reference = ReferenceRecord("t1", "ref", 0.8, "seed")
    store.append(reference)
    candidate = CandidateRecord("t1", "candidate", 0.9, True)
    decision = decide_active_grpo(reference, candidate)
    updated = maybe_update_reference(store, reference, candidate, decision)
    assert updated is not None
    assert updated.reference_answer == "candidate"


def test_unverified_candidate_above_reference_is_rejected():
    reference = ReferenceRecord("t1", "ref", 0.8, "seed")
    candidate = CandidateRecord("t1", "candidate", 0.9, False)
    decision = decide_active_grpo(reference, candidate, require_verified=True)
    assert decision.mode == "reject_candidate"
    assert not decision.should_update_reference


def test_margin_prevents_churn():
    reference = ReferenceRecord("t1", "ref", 0.8, "seed")
    candidate = CandidateRecord("t1", "candidate", 0.81, True)
    decision = decide_active_grpo(reference, candidate, margin=0.05)
    assert decision.mode == "imitate_reference"


def test_lineage_preserved_after_update(tmp_path):
    store = JSONLReferenceStore(tmp_path / "refs.jsonl")
    first = ReferenceRecord("t1", "ref", 0.8, "seed", lineage=["old"])
    store.append(first)
    candidate = CandidateRecord("t1", "candidate", 0.9, True)
    decision = decide_active_grpo(first, candidate)
    updated = maybe_update_reference(store, first, candidate, decision)
    assert updated is not None
    assert updated.lineage == ["old", "ref"]


def test_jsonl_store_reloads_correctly(tmp_path):
    store = JSONLReferenceStore(tmp_path / "refs.jsonl")
    store.append(ReferenceRecord("t1", "ref", 0.8, "seed"))
    reloaded = store.load_all()
    assert reloaded[0].task_id == "t1"
