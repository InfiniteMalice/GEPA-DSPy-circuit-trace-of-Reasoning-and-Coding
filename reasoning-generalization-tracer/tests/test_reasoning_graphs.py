import json

from rg_tracer.reasoning_graphs import (
    ReasoningEdge,
    ReasoningGraph,
    ReasoningNode,
    parse_public_reasoning_phases,
    validate_graph,
)


def _valid_graph():
    return ReasoningGraph(
        task_id="task-1",
        answer_ref="answer",
        nodes=[
            ReasoningNode("e1", "Evidence", "evidence", "observed support"),
            ReasoningNode("c1", "Claim", "claim", "answer follows"),
        ],
        edges=[ReasoningEdge("e1", "c1", "supports")],
    )


def test_valid_graph_json_parses_from_public_section():
    graph = _valid_graph()
    result = parse_public_reasoning_phases(f"<graph_json>{graph.to_json()}</graph_json>")
    assert result.graph is not None
    assert validate_graph(result.graph).valid


def test_missing_graph_section_returns_neutral_record():
    result = parse_public_reasoning_phases("plain answer")
    assert result.sections == {}
    assert result.graph is None
    assert result.diagnostics == []


def test_malformed_graph_json_returns_diagnostic():
    result = parse_public_reasoning_phases("<graph_json>{bad</graph_json>")
    assert result.graph is None
    assert result.diagnostics


def test_unknown_relation_type_is_allowed_when_nonempty():
    graph = _valid_graph()
    graph.edges[0].relation = "new_relation"
    result = validate_graph(graph, final_answer_task=False)
    assert result.valid


def test_duplicate_node_id_is_invalid():
    graph = _valid_graph()
    graph.nodes.append(ReasoningNode("c1", "Duplicate", "claim", "duplicate"))
    assert not validate_graph(graph).valid


def test_missing_edge_endpoint_is_invalid():
    graph = _valid_graph()
    graph.edges.append(ReasoningEdge("missing", "c1", "supports"))
    assert not validate_graph(graph).valid


def test_final_answer_requires_inbound_support_to_final_node():
    graph = ReasoningGraph(
        nodes=[
            ReasoningNode("e1", "Evidence", "evidence", "unconnected evidence"),
            ReasoningNode("c1", "Claim", "claim", "answer follows"),
            ReasoningNode("x1", "Side", "concept", "side node"),
        ],
        edges=[ReasoningEdge("e1", "x1", "supports")],
    )
    assert not validate_graph(graph).valid


def test_contradictory_edge_surfaces_warning():
    graph = _valid_graph()
    graph.edges.append(ReasoningEdge("c1", "e1", "contradicts"))
    result = validate_graph(graph)
    assert result.valid
    assert any("contradictory" in warning for warning in result.warnings)


def test_contradicting_evidence_does_not_count_as_support():
    graph = _valid_graph()
    graph.edges = [ReasoningEdge("e1", "c1", "contradicts")]
    result = validate_graph(graph)
    assert not result.valid
    assert any("contradictory" in warning for warning in result.warnings)


def test_graph_serialization_round_trip():
    graph = _valid_graph()
    reloaded = ReasoningGraph.from_json(json.dumps(graph.to_dict()))
    assert reloaded.to_dict() == graph.to_dict()
