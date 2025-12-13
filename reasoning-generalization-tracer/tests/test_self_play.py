import json
from pathlib import Path

from rg_tracer.concepts import ConceptSpec
from rg_tracer.runners.self_play import run_self_play


def test_self_play_creates_artifacts(tmp_path):
    problem_path = Path(__file__).resolve().parents[1] / "datasets" / "toy_math"
    problem_path = problem_path / "addition_small.jsonl"
    concept = ConceptSpec(
        name="parity", definition="Parity detector", expected_substructures=["parity"]
    )
    result = run_self_play(
        problem_path,
        profile="proof_math",
        k=3,
        concept=concept,
        output_dir=tmp_path,
    )
    run_dir = Path(result["run_dir"])
    assert (run_dir / "scores.jsonl").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "semantics.jsonl").exists()
    assert (run_dir / "attr_metrics.jsonl").exists()
    attr_dir = run_dir / "attr"
    assert attr_dir.exists()
    json_files = list(attr_dir.glob("*.json"))
    assert json_files, "attr directory should contain at least one JSON file"
    with open(run_dir / "scores.jsonl", "r", encoding="utf8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert any(candidate["passes_gates"] for candidate in lines)
    assert all("reward_case" in candidate for candidate in lines)
    assert all("thought_scores" in candidate for candidate in lines)
    semantics_text = (run_dir / "semantics.jsonl").read_text(encoding="utf8")
    semantics = [json.loads(line) for line in semantics_text.splitlines() if line.strip()]
    assert semantics and all("score" in entry for entry in semantics)
    assert all("s_match" in entry and "s_epistemic" in entry for entry in semantics)
    summary = (run_dir / "summary.md").read_text(encoding="utf8")
    assert "Attribution Metrics" in summary
    assert result["frontier"], "Pareto frontier should not be empty"
