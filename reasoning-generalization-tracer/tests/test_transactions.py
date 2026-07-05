from rg_tracer.transactions import Proposal, TransitionLog, admit_proposal


def test_valid_proposal_is_admitted():
    proposal = Proposal("p1", "t1", "answer", {"answer": "yes"}, metadata={"score": 1.0})
    decision = admit_proposal(proposal, config={"score_threshold": 0.5})
    assert decision.accepted


def test_invalid_proposal_rejected_with_reason():
    proposal = Proposal("p1", "t1", "answer", {}, metadata={"score": 0.1})
    decision = admit_proposal(proposal, config={"score_threshold": 0.5})
    assert not decision.accepted
    assert decision.violated_constraints


def test_unknown_severity_is_rejected():
    proposal = Proposal(
        "p1",
        "t1",
        "answer",
        {},
        metadata={"contradiction_severity": "typo"},
    )
    decision = admit_proposal(proposal)
    assert not decision.accepted
    assert any("unknown severity" in reason for reason in decision.violated_constraints)


def test_rejected_proposal_is_logged(tmp_path):
    proposal = Proposal("p1", "t1", "answer", {}, metadata={"score": 0.1})
    decision = admit_proposal(proposal, config={"score_threshold": 0.5})
    log = TransitionLog(tmp_path / "transitions.jsonl")
    log.append(proposal, decision)
    assert log.load()[0]["decision"]["accepted"] is False


def test_accepted_proposal_appears_in_effective_state(tmp_path):
    proposal = Proposal("p1", "t1", "answer", {"answer": "yes"})
    decision = admit_proposal(proposal)
    log = TransitionLog(tmp_path / "transitions.jsonl")
    log.append(proposal, decision)
    assert log.effective_state()["t1"]["proposal_id"] == "p1"


def test_old_state_is_not_overwritten_destructively(tmp_path):
    log = TransitionLog(tmp_path / "transitions.jsonl")
    first = Proposal("p1", "t1", "answer", {"answer": "old"})
    second = Proposal("p2", "t1", "answer", {"answer": "new"})
    log.append(first, admit_proposal(first))
    log.append(second, admit_proposal(second))
    assert len(log.load()) == 2
    assert log.effective_state()["t1"]["proposal_id"] == "p2"
