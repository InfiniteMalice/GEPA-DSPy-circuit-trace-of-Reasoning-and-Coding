"""The explicit maintainer sync command is tested with local fixture repositories."""

import importlib.util
from importlib.resources import files
from pathlib import Path

import yaml
import pytest


def _sync_module():
    path = Path(__file__).parents[1] / "scripts" / "check_17case_upstream_sync.py"
    spec = importlib.util.spec_from_file_location("check_v5_sync", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_offline_sync_statuses(tmp_path, monkeypatch):
    module = _sync_module()
    source = files("rg_tracer.epistemic_cases")
    upstream = tmp_path / "upstream" / "evaluation" / "cases"
    upstream.mkdir(parents=True)
    for name in ("17_case_manifest.yaml", "robustness_stripes.yaml"):
        upstream.joinpath(name).write_bytes(source.joinpath(name).read_bytes())
    assert module.check(tmp_path / "upstream") == "identical"
    manifest_path = upstream / "17_case_manifest.yaml"
    original = manifest_path.read_bytes()
    manifest_path.write_bytes(original + b"\n# Upstream documentation changed.\n")
    assert module.check(tmp_path / "upstream") == "upstream changed"
    manifest_path.write_bytes(original)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["cases"][0]["title"] = "Changed upstream title"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf8")
    assert module.check(tmp_path / "upstream") == "upstream changed"
    manifest["framework_version"] = "17case-v6"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf8")
    assert module.check(tmp_path / "upstream") == "incompatible version"
    manifest_path.write_text("[]", encoding="utf8")
    with pytest.raises(ValueError, match="YAML mappings"):
        module.check(tmp_path / "upstream")
    local = tmp_path / "local"
    local.mkdir()
    for name in ("17_case_manifest.yaml", "robustness_stripes.yaml", "upstream_metadata.yaml"):
        local.joinpath(name).write_bytes(source.joinpath(name).read_bytes())
    local.joinpath("robustness_stripes.yaml").write_text("modified", encoding="utf8")
    monkeypatch.setattr(module, "files", lambda package: local)
    assert module.check(tmp_path / "upstream") == "local modified"
