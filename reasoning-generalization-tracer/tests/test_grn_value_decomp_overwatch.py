import json
from pathlib import Path

import pytest

from rg_tracer.abstention import apply_abstention
from rg_tracer.modules.grn import apply_grn
from rg_tracer.modules.torch_stub import torch
from rg_tracer.runners.overwatch import OverwatchAgent, OverwatchConfig
from rg_tracer.runners.self_play import run_self_play
from rg_tracer.value_decomp import (
    compute_dvgr,
    decompose_score,
    parse_user_deep_values,
    parse_user_shallow_prefs,
)


def test_apply_grn_normalises_rms():
    tensor = torch.tensor([[3.0, 4.0]])
    normalised = apply_grn(tensor)
    values = normalised.tolist()[0]
    rms = (sum(value * value for value in values) / len(values)) ** 0.5
    assert normalised.tolist()
    assert abs(rms - 1.0) < 1e-6


def test_abstention_with_grn_flag():
    result = apply_abstention("answer", 0.5, 1.0, True, use_grn=True)
    assert result.abstained


def test_grn_affects_abstention_decision():
    # These values are chosen so GRN normalization moves the threshold crossing
    without_grn = apply_abstention("answer", 0.8, 2.5, True, use_grn=False)
    with_grn = apply_abstention("answer", 0.8, 2.5, True, use_grn=True)
    assert without_grn.abstained != with_grn.abstained


def test_value_decomposition_components():
    prompt = "Be correct, safe, and brief in style."
    deep = parse_user_deep_values(prompt)
    shallow = parse_user_shallow_prefs(prompt)
    assert deep.correctness == 1.0
    assert shallow.brevity == 1.0
    score_vector = {"logical_validity": 3.0, "rigor": 2.0}
    dvgr = compute_dvgr(
        [{"deep_value": "correct", "shallow_feature": "verbose"}],
        ["A correct solution with short form"],
    )
    assert dvgr >= 0.5
    decomposition = decompose_score(score_vector, deep, shallow, use_grn=True)
    assert decomposition["score_scalar"] > 0


def test_overwatch_rewrite_and_limits():
    rewrite_config = OverwatchConfig(
        enabled=True,
        allowed_actions=["rewrite_thought", "rewrite_action", "abort_episode", "allow"],
    )
    agent = OverwatchAgent(
        rewrite_config,
        llm=lambda _prompt: json.dumps(
            {"action": "rewrite_thought", "new_thought": "Aligned thought", "reason": "test"}
        ),
    )
    step_decision = agent.review_step([{"prompt": "p"}], {}, {})
    assert step_decision.new_thought == "Aligned thought"
    abort_agent = OverwatchAgent(
        OverwatchConfig(
            enabled=True,
            intervene_on=["unsafe"],
            allowed_actions=["rewrite_action", "abort_episode", "allow"],
            max_interventions_per_episode=1,
        ),
        # LLM returns non-JSON, testing fallback behavior
        llm=lambda _prompt: "unsafe sequence detected",
    )
    first = abort_agent.review_step([{"prompt": "p"}], {}, {})
    assert first.action == "rewrite_action"
    second = abort_agent.review_final([{"prompt": "p"}], {}, {})
    assert second.action == "allow"


def test_overwatch_disabled_short_circuits_llm():
    def _should_not_run(_prompt: str) -> str:  # pragma: no cover - defensive
        raise AssertionError("LLM should not be called when overwatch is disabled")

    agent = OverwatchAgent(OverwatchConfig(enabled=False), llm=_should_not_run)
    decision = agent.review_step([{"prompt": "p"}], {}, {})
    assert decision.action == "allow"
    assert decision.reason == "Overwatch disabled"


def test_self_play_with_grn_and_value_decomposition(tmp_path):
    problem_path = (
        Path(__file__).resolve().parents[1] / "datasets" / "toy_math" / "addition_small.jsonl"
    )
    if not problem_path.exists():
        pytest.skip(f"Test dataset not found: {problem_path}")
    overwatch_cfg = OverwatchConfig(
        enabled=True,
        allowed_actions=["observe", "rewrite_thought", "rewrite_action", "abort_episode"],
    )
    result = run_self_play(
        problem_path,
        profile="proof_math",
        k=2,
        output_dir=tmp_path,
        value_decomp_enabled=True,
        use_grn_for_scoring=True,
        use_grn_for_abstention=True,
        overwatch_config=overwatch_cfg,
    )
    run_dir = Path(result["run_dir"])
    scores = [
        json.loads(line)
        for line in (run_dir / "scores.jsonl").read_text(encoding="utf8").splitlines()
        if line.strip()
    ]
    assert scores and "grn_flags" in scores[0]
    assert "value_decomp_user_deep" in scores[0]
