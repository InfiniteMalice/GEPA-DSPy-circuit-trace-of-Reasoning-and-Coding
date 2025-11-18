from rg_tracer.fallback.academic_pipeline import _iter_tag_labels


def test_iter_tag_labels_handles_basic_iterables():
    entry = {"tags": ["a", "b"]}
    assert list(_iter_tag_labels(entry)) == ["a", "b"]


def test_iter_tag_labels_handles_nested_mapping():
    entry = {"tags": {"tags": ("x", "y")}}
    assert list(_iter_tag_labels(entry)) == ["x", "y"]


def test_iter_tag_labels_handles_scalar():
    entry = {"tags": "solo"}
    assert list(_iter_tag_labels(entry)) == ["solo"]


def test_iter_tag_labels_handles_malformed_payloads():
    entry = {"tags": 3.14}
    assert list(_iter_tag_labels(entry)) == []
    entry = {}
    assert list(_iter_tag_labels(entry)) == []
