from rg_tracer.humanities import analyse_humanities_chain, load_profiles, evaluate_profile


def test_humanities_profiles_and_signals():
    steps = [
        "Quote: 'evidence' (Smith 1908, p.41).",
        "However critics note rural gaps while balance is discussed.",
        "Perhaps reforms worked in cities but not everywhere.",
    ]
    signals = analyse_humanities_chain(steps)
    assert signals.citation_coverage > 0
    assert signals.hedge_rate > 0
    profiles = load_profiles()
    profile = profiles["humanities"]
    metrics = {
        "source_handling": {"positive": 0.9, "coverage": 0.9},
        "interpretive_fidelity": {"positive": 0.9, "coverage": 0.8},
        "historiography_context": {"positive": 0.8, "coverage": 0.7},
        "causal_discipline": {"positive": 0.9, "coverage": 0.8},
        "triangulation": {"positive": 0.8, "coverage": 0.7},
        "normative_positive_sep": {"positive": 0.8, "coverage": 0.8},
        "uncertainty_calibration": {"positive": 0.8, "coverage": 0.9},
        "intellectual_charity": {"positive": 0.75, "coverage": 0.7},
        "rhetorical_hygiene": {"positive": 0.9, "coverage": 0.9},
        "reproducibility_transparency": {"positive": 0.8, "coverage": 0.8},
        "synthesis_generalization": {"positive": 0.85, "coverage": 0.8},
        "epistemic_neutrality": {"positive": 0.8, "coverage": 0.8},
    }
    result = evaluate_profile(metrics, profile)
    assert result["passes_gates"]
    metrics["source_handling"]["positive"] = 0.2
    result_low = evaluate_profile(metrics, profile)
    assert not result_low["passes_gates"]
