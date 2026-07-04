from rg_tracer.claim_validation import (
    RepairTrace,
    ValidationContract,
    check_claim_drift,
    validate_contract,
)


def test_valid_contract_passes():
    contract = ValidationContract("c1", "t1", "claim", ["paper"], ["pytest"])
    result = validate_contract(contract, {"evidence": ["paper"], "tests": ["pytest"]})
    assert result["passed"]


def test_missing_required_evidence_fails():
    contract = ValidationContract("c1", "t1", "claim", ["paper"], [])
    result = validate_contract(contract, {"evidence": [], "tests": []})
    assert not result["passed"]
    assert result["missing_evidence"] == ["paper"]


def test_unsupported_claim_emits_drift_report():
    report = check_claim_drift("t1", "claim", "artifact", {"unsupported_claims": ["claim"]})
    assert report.drift_type == "unsupported_claim"
    assert report.severity == "high"


def test_repair_trace_serializes_and_reloads():
    trace = RepairTrace("t1", "a1", "failed", "fixed", ["pytest"], "passed")
    reloaded = RepairTrace.from_json(trace.to_json())
    assert reloaded.to_dict() == trace.to_dict()
