from rg_tracer.concepts.circuit_adapter import CircuitTrace, trace_model


def test_trace_model_returns_normalised_trace():
    trace = trace_model("demo", "task1")
    assert isinstance(trace, CircuitTrace)
    payload = trace.to_json()
    assert payload["model_ref"] == "demo"
    assert "features" in payload and payload["features"]
