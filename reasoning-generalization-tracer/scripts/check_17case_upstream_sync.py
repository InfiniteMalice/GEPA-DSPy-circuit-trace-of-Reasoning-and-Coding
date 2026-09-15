"""Explicit read-only comparison of the pinned V5 mirror with upstream main."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from urllib.request import urlopen

import yaml


def check(upstream_directory: Path | None = None) -> str:
    """Return identical, local modified, upstream changed, or incompatible version.

    A supplied directory enables an entirely offline maintainer comparison.
    Network mode resolves main once, then reads both files at the same commit.
    """
    root = files("rg_tracer.epistemic_cases")
    metadata = yaml.safe_load(root.joinpath("upstream_metadata.yaml").read_text(encoding="utf8"))
    local = {}
    for name, source in metadata["sources"].items():
        content = root.joinpath(name).read_bytes()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            return "local modified"
        local[name] = content
    repository = metadata["upstream_repository"]
    if upstream_directory is None:
        with urlopen(
            f"https://api.github.com/repos/{repository}/commits/main", timeout=30
        ) as reply:
            commit = json.load(reply)["sha"]
    remote = {}
    for name, source in metadata["sources"].items():
        path = source["upstream_path"]
        if upstream_directory is None:
            with urlopen(
                f"https://raw.githubusercontent.com/{repository}/{commit}/{path}", timeout=30
            ) as reply:
                content = reply.read()
        else:
            content = (upstream_directory / path).read_bytes()
        remote[name] = content
    manifest = yaml.safe_load(remote["17_case_manifest.yaml"])
    stripes = yaml.safe_load(remote["robustness_stripes.yaml"])
    if not isinstance(manifest, dict) or not isinstance(stripes, dict):
        raise ValueError("Upstream contract documents must be YAML mappings")
    if (
        manifest.get("framework_version") != "17case-v5"
        or stripes.get("registry_version") != "17case-v5"
    ):
        return "incompatible version"
    return "identical" if local == remote else "upstream changed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-directory", type=Path)
    args = parser.parse_args()
    try:
        status = check(args.upstream_directory)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        parser.exit(2, f"Sync check failed: {exc}\n")
    print(status)
    return 0 if status == "identical" else 1


if __name__ == "__main__":
    raise SystemExit(main())
