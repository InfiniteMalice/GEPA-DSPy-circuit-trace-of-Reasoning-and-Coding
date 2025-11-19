from rg_tracer.fallback.academic_pipeline import _iter_tag_labels


def test_iter_tag_labels_handles_basic_iterables() -> None:
    entry = {"tags": ["a", "b"]}
    assert list(_iter_tag_labels(entry)) == ["a", "b"]


def test_iter_tag_labels_handles_scalar() -> None:
    entry = {"tags": "solo"}
    assert list(_iter_tag_labels(entry)) == ["solo"]


def test_iter_tag_labels_handles_malformed_payloads() -> None:
    entry = {"tags": 3.14}
    assert list(_iter_tag_labels(entry)) == []
    entry = {}
    assert list(_iter_tag_labels(entry)) == []


def test_iter_tag_labels_filters_non_strings_in_iterables() -> None:
    entry = {"tags": ["a", 123, "b", None]}
    assert list(_iter_tag_labels(entry)) == ["a", "b"]


def test_iter_tag_labels_handles_empty_iterables() -> None:
    entry = {"tags": []}
    assert list(_iter_tag_labels(entry)) == []


def test_iter_tag_labels_supports_other_iterables() -> None:
    entry = {"tags": ("x", "y")}
    assert list(_iter_tag_labels(entry)) == ["x", "y"]
    entry = {"tags": {"alpha", "beta"}}
    assert sorted(_iter_tag_labels(entry)) == ["alpha", "beta"]


def test_iter_tag_labels_handles_nested_mapping_scalar() -> None:
    entry = {"tags": {"tags": "inner"}}
    assert list(_iter_tag_labels(entry)) == ["inner"]


def test_iter_tag_labels_handles_nested_mapping_iterable() -> None:
    entry = {"tags": {"tags": ["first", "second"]}}
    assert list(_iter_tag_labels(entry)) == ["first", "second"]
