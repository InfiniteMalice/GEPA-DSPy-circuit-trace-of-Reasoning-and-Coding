"""The explicit maintainer sync command is tested with local fixture repositories."""

import importlib.util
import io
import json
from importlib.resources import files
from pathlib import Path

import yaml
import pytest


def _sync_module():
    """Load the maintainer command without running its CLI entry point."""
    path = Path(__file__).parents[1] / "scripts" / "check_17case_upstream_sync.py"
    spec = importlib.util.spec_from_file_location("check_v5_sync", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_offline_sync_statuses(tmp_path, monkeypatch):
    """Offline comparisons report byte/version drift without opening the network."""
    module = _sync_module()

    def no_network(*args, **kwargs):
        """Fail if offline comparison attempts any HTTP request."""
        raise AssertionError("Offline sync attempted network access")

    monkeypatch.setattr(module, "urlopen", no_network)
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


@pytest.mark.parametrize(
    "scenario",
    ["identical", "drift", "pinned_mismatch", "pinned_error", "incompatible", "main_error"],
)
@pytest.mark.parametrize("affected_name", ["17_case_manifest.yaml", "robustness_stripes.yaml"])
def test_network_verifies_pinned_resources_before_main(monkeypatch, scenario, affected_name):
    """A matching main cannot mask incorrect or unavailable pinned provenance."""
    module = _sync_module()
    root = files("rg_tracer.epistemic_cases")
    metadata = yaml.safe_load(root.joinpath("upstream_metadata.yaml").read_text())
    repository = metadata["upstream_repository"]
    pinned = metadata["upstream_commit"]
    main = "b" * 40
    requests = []
    pinned_urls = [
        f"https://raw.githubusercontent.com/{repository}/{pinned}/{source['upstream_path']}"
        for source in metadata["sources"].values()
    ]
    main_url = f"https://api.github.com/repos/{repository}/commits/main"

    def fetch(url, timeout):
        """Serve pinned and main revisions with independently controlled failures."""
        requests.append(url)
        assert timeout == 30
        if url == main_url:
            if scenario == "main_error":
                raise OSError("main unavailable")
            return io.BytesIO(json.dumps({"sha": main}).encode())
        for name, source in metadata["sources"].items():
            if url.endswith("/" + source["upstream_path"]):
                content = root.joinpath(name).read_bytes()
                if f"/{pinned}/" in url and name == affected_name:
                    if scenario == "pinned_error":
                        raise OSError("pinned commit not found")
                    if scenario == "pinned_mismatch":
                        content += b"\n# Different pinned revision\n"
                elif f"/{main}/" in url and scenario == "drift":
                    content += b"\n# Main changed\n"
                elif f"/{main}/" in url and scenario == "incompatible":
                    content = content.replace(b"17case-v5", b"17case-v6")
                return io.BytesIO(content)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(module, "urlopen", fetch)
    if scenario.startswith("pinned_"):
        with pytest.raises(ValueError, match="Pinned upstream"):
            module.check()
        failed_index = list(metadata["sources"]).index(affected_name)
        assert requests == pinned_urls[: failed_index + 1]
    elif scenario == "main_error":
        with pytest.raises(OSError, match="main unavailable"):
            module.check()
        assert requests == pinned_urls + [main_url]
    else:
        expected = {
            "identical": "identical",
            "drift": "upstream changed",
            "incompatible": "incompatible version",
        }
        assert module.check() == expected[scenario]
        assert requests[:3] == pinned_urls + [main_url]
        assert all(f"/{main}/" in url for url in requests[3:])
