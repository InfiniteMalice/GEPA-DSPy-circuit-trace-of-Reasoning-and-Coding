from rg_tracer.scoring import axes


def test_logical_validity_scores_formal_proof():
    metrics = {"formal_proof": True, "contradictions": 0}
    assert axes.logical_validity(metrics) == 4


def test_numerical_accuracy_boundary():
    metrics = {"error_rate": 0.15, "error_tolerance": 0.1}
    assert axes.numerical_accuracy(metrics) == 2


def test_abstraction_generalization_bonus():
    metrics = {"transfer_accuracy": 0.9, "compression_gain": 0.2, "variable_lifts": 2, "theorem_induced": 3}
    assert axes.abstraction_generalization(metrics) == 4
