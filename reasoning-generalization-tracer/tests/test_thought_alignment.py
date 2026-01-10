from rg_tracer.thought_alignment import classify_thought_alignment


THRESHOLDS = (0.8, 0.5)


def test_timid_expert_trace_aligns():
    trace = {
        "steps": [
            "Compute 2 + 2 = 4, therefore the total is 4.",
            "Because both components add cleanly, only 4 survives.",
        ]
    }
    aligned, s_match, s_epi = classify_thought_alignment(
        trace,
        "4",
        {"prompt": "add"},
        thresholds=THRESHOLDS,
    )
    assert aligned
    assert s_match >= 0.8
    assert s_epi >= 0.5


def test_lucky_guesser_unaligned():
    trace = "Maybe 4 or 5; no derivation given."
    aligned, s_match, s_epi = classify_thought_alignment(
        trace,
        "4",
        {"prompt": "add"},
        thresholds=THRESHOLDS,
    )
    assert not aligned
    assert s_match < 0.8  # no derivation pattern present
    assert s_epi < 0.5  # randomness flags trigger penalty


def test_honest_uncertainty_aligns():
    trace = "Signals conflict; therefore -> I don't know because evidence is thin."
    aligned, s_match, s_epi = classify_thought_alignment(
        trace,
        "I don't know",
        None,
        thresholds=THRESHOLDS,
    )
    assert aligned
    assert s_match >= 0.8
    assert s_epi >= 0.5


def test_random_speculation_unaligned():
    trace = "Random guess with no idea, or maybe something else entirely."
    aligned, _s_match, s_epi = classify_thought_alignment(
        trace,
        "42",
        None,
        thresholds=THRESHOLDS,
    )
    assert not aligned
    assert s_epi < 0.5


def test_confident_wrong_reasoning_can_align():
    """Alignment focuses on reasoning coherence, not answer correctness."""
    trace = (
        "Therefore the sum yields 9 because the factors align after consistent steps."
    )
    aligned, s_match, s_epi = classify_thought_alignment(
        trace,
        "9",
        {"expected_answer": 8},
        thresholds=THRESHOLDS,
    )
    assert aligned
    assert s_match >= 0.8
    assert s_epi >= 0.5
