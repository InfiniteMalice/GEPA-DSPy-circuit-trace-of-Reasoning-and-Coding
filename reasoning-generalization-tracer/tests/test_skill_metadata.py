import pytest

from rg_tracer.skills import (
    SkillDependency,
    SkillManifest,
    dependency_graph,
    recursive_skill_warnings,
    skill_risk_warnings,
)


def test_manifest_round_trip():
    manifest = SkillManifest(
        "skill-a",
        "Skill A",
        "Does work",
        "1.0.0",
        "run",
        [SkillDependency("package", "dep", "1.0", "pypi")],
        "local",
    )
    assert SkillManifest.from_json(manifest.to_json()).to_dict() == manifest.to_dict()


def test_missing_provenance_warning():
    manifest = SkillManifest("skill-a", "Skill A", "", "1.0.0", "run")
    assert "missing provenance" in skill_risk_warnings(manifest)


def test_external_service_warning():
    manifest = SkillManifest(
        "skill-a",
        "Skill A",
        "",
        "1.0.0",
        "run",
        [SkillDependency("service", "api", "1.0", "https://example.invalid")],
        "local",
    )
    assert any("external service" in warning for warning in skill_risk_warnings(manifest))


def test_dependency_graph_generation():
    manifest = SkillManifest(
        "skill-a",
        "Skill A",
        "",
        "1.0.0",
        "run",
        [SkillDependency("skill", "skill-b", "1.0", "local")],
        "local",
    )
    assert dependency_graph([manifest]) == {"skill-a": ["skill-b"]}


def test_dependency_graph_rejects_duplicate_skill_ids():
    first = SkillManifest("skill-a", "Skill A", "", "1.0.0", "run", [], "local")
    second = SkillManifest("skill-a", "Skill A2", "", "1.0.0", "run", [], "local")
    with pytest.raises(ValueError, match="duplicate skill_id"):
        dependency_graph([first, second])


def test_recursive_skill_warnings_detect_cycle():
    first = SkillManifest(
        "skill-a",
        "Skill A",
        "",
        "1.0.0",
        "run",
        [SkillDependency("skill", "skill-b", "1.0", "local")],
        "local",
    )
    second = SkillManifest(
        "skill-b",
        "Skill B",
        "",
        "1.0.0",
        "run",
        [SkillDependency("skill", "skill-a", "1.0", "local")],
        "local",
    )
    warnings = recursive_skill_warnings([first, second])
    assert any("recursive skill dependency" in warning for warning in warnings)
