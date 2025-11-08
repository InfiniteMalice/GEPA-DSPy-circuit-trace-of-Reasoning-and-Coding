from rg_tracer.semantics import SemanticTag, repair_once, verify_chain


def test_semantics_detects_and_repairs():
    chain = "Let y = 2.\nTherefore the result is 5 meters."
    spec = {"concept": "carry", "units": "count", "variables": ["x"]}
    report = verify_chain(chain, spec)
    assert report.score < 2
    assert any(SemanticTag.VARIABLE_DRIFT.value in entry["tags"] for entry in report.tags)
    repaired_steps = repair_once(
        chain,
        report.tags,
        expected_units="count",
        preferred_variables=["x"],
    )
    repaired_report = verify_chain("\n".join(repaired_steps), spec)
    assert repaired_report.score >= report.score
    assert repaired_report.unit_check_pass
