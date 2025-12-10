from rg_tracer.trm_baseline import (
    TinyRecursionModel,
    evaluate,
    generate_parity_data,
    train,
)


def test_trm_training_improves_accuracy():
    data = generate_parity_data(32, length=4)
    model = TinyRecursionModel(learning_rate=0.02)
    baseline = evaluate(model, data)
    train(model, data, epochs=8)
    improved = evaluate(model, data)
    assert improved.accuracy >= baseline.accuracy
    assert len(improved.semantic_scores) == len(data)
    assert all(score >= 0 for score in improved.semantic_scores)
