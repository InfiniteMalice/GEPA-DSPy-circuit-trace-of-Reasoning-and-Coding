import json

from rg_tracer.fallback import run_academic_pipeline


def test_academic_pipeline_outputs_bayesian_position(tmp_path):
    problem_path = tmp_path / "claims.jsonl"
    records = [
        {
            "id": "case-1",
            "claim": "Reforms lowered mortality",
            "concept": "health",
            "analysis": [
                "Reports cite a 20% decline (Lee 1920, p.5) and likely reflect reform scope.",
                "However, rural clinics contest the drop and both sides are summarised.",
            ],
            "prior": 0.6,
            "evidence": [
                {
                    "source": "Lee 1920",
                    "year": 1920,
                    "method": "archival",
                    "finding": "Urban hospitals record declines",
                    "limitations": "Urban sample",
                    "support_if_true": 0.75,
                    "support_if_false": 0.45,
                }
            ],
        },
        {
            "id": "case-2",
            "claim": "Parades guaranteed unity",
            "concept": "culture",
            "analysis": [
                "Speeches insist success without cites.",
                "Therefore unity followed.",
            ],
            "prior": 0.5,
            "evidence": [],
        },
    ]
    problem_path.write_text("\n".join(json.dumps(record) for record in records))
    output = run_academic_pipeline(problem_path)
    summary = output["summary"]
    assert summary["count"] == 2
    records = output["records"]
    confident = [record for record in records if not record["abstained"]]
    assert confident and "posterior" in confident[0]
    abstained = [record for record in records if record["abstained"]]
    assert abstained and abstained[0]["decision"] == "I don't know."
    table = confident[0]["evidence_table"]
    assert table and table[0]["source"] == "Lee 1920"
    assert confident[0]["decision_policy"] in {
        "Support with caveats",
        "Gather more evidence",
    }
