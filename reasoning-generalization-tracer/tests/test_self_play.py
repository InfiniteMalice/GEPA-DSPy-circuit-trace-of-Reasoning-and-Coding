import json
from pathlib import Path

from rg_tracer.concepts import ConceptSpec
from rg_tracer.runners.self_play import run_self_play


def test_self_play_creates_artifacts(tmp_path):
    problem_path = Path(__file__).resolve().parents[1] / "src" / "rg_tracer" / "datasets" / "toy_math" / "addition_small.jsonl"
    concept = ConceptSpec(name="parity", definition="Parity detector", expected_substructures=["parity"])
    result = run_self_play(problem_path, profile="proof_math", k=3, concept=concept, output_dir=tmp_path)
    run_dir = Path(result["run_dir"])
    assert (run_dir / "scores.jsonl").exists()
    assert (run_dir / "summary.md").exists()
    with open(run_dir / "scores.jsonl", "r", encoding="utf8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert any(candidate["passes_gates"] for candidate in lines)
    assert result["frontier"], "Pareto frontier should not be empty"
