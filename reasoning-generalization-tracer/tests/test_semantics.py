from rg_tracer.semantics import SemanticTag, repair_once, verify_chain


def test_semantics_detects_and_repairs():
    chain = "Let y = 2.\nTherefore the result is 5 meters."
    spec = {"concept": "carry", "units": "count", "variables": ["x"]}
    report = verify_chain(chain, spec)
    assert report.score < 2
    assert any(SemanticTag.VARIABLE_DRIFT.value in entry.get("tags", ()) for entry in report.tags)
    assert any(SemanticTag.UNIT_MISMATCH.value in entry.get("tags", ()) for entry in report.tags)
    repaired_steps = repair_once(
        chain,
        report.tags,
        expected_units="count",
        preferred_variables=["x"],
    )
    repaired_report = verify_chain("\n".join(repaired_steps), spec)
    assert repaired_report.score >= report.score
    assert repaired_report.unit_check_pass


def test_semantics_ignores_numeric_variables():
    chain = "Let x = 1."
    spec = {"concept": "carry", "variables": [1, 2]}
    report = verify_chain(chain, spec)
    assert not any(
        SemanticTag.VARIABLE_DRIFT.value in entry.get("tags", ()) for entry in report.tags
    )


def test_semantics_flags_unlisted_variables_with_mixed_types():
    chain = "Let x = 1. Then y = 2."
    spec = {"concept": "carry", "variables": ["x", 1, b"v"]}
    report = verify_chain(chain, spec)
    assert any(SemanticTag.VARIABLE_DRIFT.value in entry.get("tags", ()) for entry in report.tags)
    offending = [
        entry.get("offending_token") for entry in report.tags if entry.get("offending_token")
    ]
    assert "y" in offending


def test_semantics_allows_empty_variable_list():
    chain = "z equals 3."
    spec = {"concept": "carry", "variables": []}
    report = verify_chain(chain, spec)
    assert not any(
        SemanticTag.VARIABLE_DRIFT.value in entry.get("tags", ()) for entry in report.tags
    )


def test_semantics_trims_variable_names():
    chain = " z appears in the result."
    spec = {"concept": "carry", "variables": ["  z  "]}
    report = verify_chain(chain, spec)
    assert not any(
        SemanticTag.VARIABLE_DRIFT.value in entry.get("tags", ()) for entry in report.tags
    )
