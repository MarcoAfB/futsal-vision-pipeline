import copy
from pathlib import Path

import pytest
import yaml

from futsal.datasets.datasetFiles import readJson, validateNewOutput
from futsal.datasets.datasetPlan import buildDatasetPlan
from futsal.datasets.datasetProjection import prepareProjection
from futsal.datasets.rawSources import importRawSources, verifyRawSources
from futsal.evaluation.evaluationData import validateResearchOutput
from futsal.inventory.catalog import hashFile, loadCatalog, resolveSourcePath
from futsal.inventory.inventory import buildInventory
from futsal.reporting.report import validateOutputDirectory


def importedSources(preparedSources):
    source, recipe, planDir = preparedSources
    catalogFile = recipe.parent / "dataSources.yaml"
    catalog = loadCatalog(catalogFile)
    destination = planDir.parent / "raw"
    manifest = importRawSources(catalog, source, destination, progress=lambda message: None)
    native = recipe.parent / "native.yaml"
    native.write_text(yaml.safe_dump(manifest["catalog"], sort_keys=False), encoding="utf-8")
    return source, destination, native, manifest


def testRawImportUsesOnlyDeclaredRealInputsAndStableRecords(preparedSources, monkeypatch):
    source, recipe, _ = preparedSources
    before = buildInventory(source, loadCatalog(recipe.parent / "dataSources.yaml"))
    (source / "unrelated.pt").write_bytes(b"not an input")
    (source / "unused.xml").write_bytes(b"not declared")
    source, destination, native, manifest = importedSources(preparedSources)
    assert manifest["sourcePolicy"] == "realOnly"
    assert manifest["catalog"]["schemaVersion"] == 2
    assert "legacyReviewPath" not in manifest["catalog"]
    assert all("legacyMatchId" not in video for video in manifest["catalog"]["ownVideos"])
    assert all(Path(item["path"]).suffix not in {".yaml", ".pt"} for item in manifest["artifacts"])
    assert not any(item["path"].endswith("unused.xml") for item in manifest["artifacts"])
    assert not (destination / "reviewed.yaml").exists()
    originalOpen = Path.open

    def isolatedOpen(path, *args, **kwargs):
        if path.is_relative_to(source):
            raise AssertionError(f"Unexpected source access: {path}")
        return originalOpen(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", isolatedOpen)
    after = buildInventory(destination, loadCatalog(native))
    assert after["summary"] == before["summary"]
    assert [row["recordId"] for video in after["ownVideos"] for row in video["frames"]] == [
        row["recordId"] for video in before["ownVideos"] for row in video["frames"]]
    assert [[row["recordId"] for row in source["samples"]] for source in after["externalSources"]] == [
        [row["recordId"] for row in source["samples"]] for source in before["externalSources"]]
    for video in before["ownVideos"]:
        assert resolveSourcePath(destination, video["videoPath"]).is_file()
    assert verifyRawSources(destination)["contractId"] == manifest["contractId"]


def testRawVerificationRejectsCorruptionAndExtraFiles(preparedSources):
    _, root, _, manifest = importedSources(preparedSources)
    target = root / manifest["artifacts"][0]["path"]
    original = target.read_bytes()
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Input changed"):
        verifyRawSources(root)
    target.write_bytes(original)
    (root / "unregistered.txt").write_text("extra")
    with pytest.raises(ValueError, match="membership"):
        verifyRawSources(root)


def testRawImportRefusesOverwriteAndInsufficientSpace(preparedSources, monkeypatch):
    source, root, native, _ = importedSources(preparedSources)
    with pytest.raises(ValueError, match="outside dataRoot"):
        importRawSources(loadCatalog(native), root, root)
    catalog = loadCatalog(native)
    before = hashFile(root / "sourceManifest.json")

    def insufficient(*args):
        raise ValueError("Insufficient disk space")

    monkeypatch.setattr("futsal.datasets.rawSources.requireDiskSpace", insufficient)
    with pytest.raises(ValueError, match="Insufficient"):
        importRawSources(catalog, root, source.parent / "noSpace")
    assert hashFile(root / "sourceManifest.json") == before
    assert not (source.parent / "noSpace").exists()


def testCurrentProjectionKeepsFrozenIdentityAfterRawRelocation(projectionBundle, preparedSources):
    parent, originalProjection, recipe, _ = projectionBundle
    source, root, native, _ = importedSources(preparedSources)
    nativeRecipe = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    nativeRecipe["catalogPath"] = native.name
    config = recipe.parent / "nativeProjection.yaml"
    config.write_text(yaml.safe_dump(nativeRecipe, sort_keys=False), encoding="utf-8")
    output = root.parent / "nativeProjection"
    result = prepareProjection(parent, config, root, output, progress=lambda message: None)
    original = readJson(originalProjection / "projection.json")
    assert result["samples"] == original["samples"]
    assert result["summary"] == original["summary"]
    assert result["parentContractId"] == original["parentContractId"]
    assert result["supplementSourceEvidence"] == original["supplementSourceEvidence"]
    changed = copy.deepcopy(yaml.safe_load(native.read_text()))
    changed["ownVideos"][0]["reviewedIntervals"][0]["endFrame"] = 0
    native.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="scientific catalogue"):
        prepareProjection(parent, config, root, root.parent / "wrongReview", progress=lambda message: None)
    assert source.is_dir()


def testNativePlanningUsesConfigurationMappingEvidence(preparedSources):
    _, raw, native, _ = importedSources(preparedSources)
    _, recipe, _ = preparedSources
    value = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    value["catalogPath"] = native.name
    target = recipe.parent / "nativeCurriculum.yaml"
    target.write_text(yaml.safe_dump(value), encoding="utf-8")
    plan = buildDatasetPlan(target, raw)
    assert plan["catalog"]["data"]["schemaVersion"] == 2
    assert all(item["path"] != "reviewed.yaml" and item["path"] != "publicEvidence.yaml" for item in plan["sourceEvidence"])
    assert plan["inventorySummary"]["reviewedFrames"] == 6


@pytest.mark.parametrize("validator", [validateNewOutput, validateOutputDirectory])
def testOtherInputRootsCannotPlaceOutputsInsideLocalRaw(validator, tmp_path):
    project = Path(__file__).resolve().parents[2]
    target = project / "data/raw/accidentalOutput"
    with pytest.raises(ValueError, match="outside dataRoot"):
        validator(target, tmp_path)
    assert not target.exists()


def testResearchOutputProtectsLocalRawTree():
    project = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="outside dataRoot"):
        validateResearchOutput(project / "data/raw/accidentalRun", [])
