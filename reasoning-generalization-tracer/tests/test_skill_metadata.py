from rg_tracer.skills import (
    SkillDependency,
    SkillManifest,
    dependency_graph,
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
