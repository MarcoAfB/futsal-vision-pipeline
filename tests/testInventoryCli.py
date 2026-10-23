import json

import pytest
import yaml

from futsal.catalog import hashFile
from futsal.cli import main
from futsal.report import validateOutputDirectory


def testCliWritesDeterministicReportsWithoutChangingSources(smallDataset, capsys):
    dataRoot, configPath, outputDir = smallDataset
    before = {
        path.relative_to(dataRoot): hashFile(path) for path in dataRoot.rglob("*") if path.is_file()
    }
    arguments = [
        "inventory",
        "--dataRoot",
        str(dataRoot),
        "--outputDir",
        str(outputDir),
        "--config",
        str(configPath),
    ]
    assert main(arguments) == 0
    inventoryPath = outputDir / "inventory.json"
    firstHash = hashFile(inventoryPath)
    assert main(arguments) == 0
    assert hashFile(inventoryPath) == firstHash
    after = {
        path.relative_to(dataRoot): hashFile(path) for path in dataRoot.rglob("*") if path.is_file()
    }
    assert after == before
    assert "Reviewed frame records: 4; matches: 1" in capsys.readouterr().out
    assert "No split has been assigned" in (outputDir / "report.md").read_text(encoding="utf-8")
    inventory = json.loads(inventoryPath.read_text(encoding="utf-8"))
    assert inventory["reviewRegistryEvidence"]["sha256"] == hashFile(dataRoot / "reviewed.yaml")
    assert str(dataRoot) not in inventoryPath.read_text(encoding="utf-8")


def testCliKeepsAuditWhenDataHasErrors(smallDataset):
    dataRoot, configPath, outputDir = smallDataset
    (dataRoot / "old.xml").write_text("not XML", encoding="utf-8")
    assert (
        main(
            [
                "inventory",
                "--dataRoot",
                str(dataRoot),
                "--outputDir",
                str(outputDir),
                "--config",
                str(configPath),
            ]
        )
        == 1
    )
    result = json.loads((outputDir / "inventory.json").read_text(encoding="utf-8"))
    assert result["status"] == "requiresReview"
    assert result["summary"]["issueCounts"]["annotationUnreadable"] == 1


def testOutputCannotBeInsideSources(smallDataset):
    dataRoot, configPath, outputDir = smallDataset
    for target in (dataRoot, dataRoot / "results"):
        with pytest.raises(ValueError, match="outside dataRoot"):
            validateOutputDirectory(target, dataRoot)
    with pytest.raises(SystemExit) as result:
        main(
            [
                "inventory",
                "--dataRoot",
                str(dataRoot),
                "--outputDir",
                str(dataRoot / "results"),
                "--config",
                str(configPath),
            ]
        )
    assert result.value.code == 2
    assert not (dataRoot / "results").exists()


def testCliInvalidCatalogDoesNotCreateReports(smallDataset):
    dataRoot, configPath, outputDir = smallDataset
    configPath.write_text("schemaVersion: 7", encoding="utf-8")
    with pytest.raises(SystemExit) as result:
        main(
            [
                "inventory",
                "--dataRoot",
                str(dataRoot),
                "--outputDir",
                str(outputDir),
                "--config",
                str(configPath),
            ]
        )
    assert result.value.code == 2
    assert not outputDir.exists()


@pytest.mark.parametrize("configTarget", ["inventory.json", "report.md"])
def testOutputsCannotOverwriteConfiguration(smallDataset, configTarget):
    dataRoot, configPath, outputDir = smallDataset
    if configTarget == "inventory.json":
        newConfig = configPath.with_name(configTarget)
        newConfig.write_bytes(configPath.read_bytes())
        configPath = newConfig
        protectedPath = newConfig
    else:
        config = yaml.safe_load(configPath.read_text())
        protectedPath = configPath.with_name(configTarget)
        protectedPath.write_bytes(configPath.with_name("labelMap.yaml").read_bytes())
        config["labelMapPath"] = configTarget
        configPath.write_text(yaml.safe_dump(config), encoding="utf-8")
    before = hashFile(protectedPath)
    with pytest.raises(SystemExit) as result:
        main(
            [
                "inventory",
                "--dataRoot",
                str(dataRoot),
                "--outputDir",
                str(configPath.parent),
                "--config",
                str(configPath),
            ]
        )
    assert result.value.code == 2
    assert hashFile(protectedPath) == before


def testBothOutputTargetsAreCheckedBeforeWriting(smallDataset):
    dataRoot, configPath, outputDir = smallDataset
    outputDir.mkdir()
    (outputDir / "report.md").mkdir()
    with pytest.raises(SystemExit) as result:
        main(
            [
                "inventory",
                "--dataRoot",
                str(dataRoot),
                "--outputDir",
                str(outputDir),
                "--config",
                str(configPath),
            ]
        )
    assert result.value.code == 2
    assert not (outputDir / "inventory.json").exists()
