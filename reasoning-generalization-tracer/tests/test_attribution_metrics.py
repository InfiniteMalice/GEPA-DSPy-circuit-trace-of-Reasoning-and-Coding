from rg_tracer.attribution import metrics

BASE_NODES = [
    {"id": "n0", "layer": 0, "type": "token", "activation": 1.0},
    {"id": "n1", "layer": 1, "type": "mlp", "activation": 0.5},
    {"id": "n2", "layer": 2, "type": "logit", "activation": 0.6},
]


def _graph(edges, phase):
    return {
        "model_ref": "m",
        "task_id": "t",
        "nodes": BASE_NODES,
        "edges": edges,
        "meta": {"phase": phase},
    }


def test_path_sparsity_monotonic():
    smooth = _graph(
        [
            {"src": "n0", "dst": "n1", "attr": 0.5},
            {"src": "n1", "dst": "n2", "attr": 0.5},
        ],
        phase="overfit",
    )
    spiky = _graph(
        [
            {"src": "n0", "dst": "n1", "attr": 0.9},
            {"src": "n1", "dst": "n2", "attr": 0.1},
        ],
        phase="overfit",
    )
    assert metrics.path_sparsity([spiky]) > metrics.path_sparsity([smooth])


def test_alignment_delta_positive():
    overfit = _graph(
        [
            {"src": "n0", "dst": "n1", "attr": 0.2},
            {"src": "n1", "dst": "n2", "attr": 0.1},
        ],
        phase="overfit",
    )
    post = _graph(
        [
            {"src": "n0", "dst": "n1", "attr": 0.3},
            {"src": "n1", "dst": "n2", "attr": 0.8},
        ],
        phase="post_grok",
    )
    concept_features = [{"id": "n2"}]
    delta = metrics.delta_alignment([overfit, post], concept_features)
    assert delta > 0


def test_repeatability_recognises_divergence():
    g1 = _graph(
        [
            {"src": "n0", "dst": "n1", "attr": 0.8},
            {"src": "n1", "dst": "n2", "attr": 0.2},
        ],
        phase="overfit",
    )
    g2 = _graph(
        [
            {"src": "n0", "dst": "n1", "attr": 0.1},
            {"src": "n1", "dst": "n2", "attr": 0.9},
        ],
        phase="overfit",
    )
    identical = metrics.repeatability([g1, g1])
    divergent = metrics.repeatability([g1, g2])
    assert divergent < identical
