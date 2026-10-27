import hashlib

import pytest
import yaml

from futsal.catalog import hashFile
from futsal.cli import main
from futsal.datasetExport import prepareDataset
from futsal.datasetFiles import hashJson, readJson, writeJson
from futsal.datasetPlan import buildDatasetPlan, writeDatasetPlan
from futsal.datasetVerify import verifyDataset


def exportedFixture(preparedSources):
    dataRoot, configPath, planDir = preparedSources
    planPath = writeDatasetPlan(buildDatasetPlan(configPath, dataRoot), planDir, dataRoot)
    output = planDir.parent / "prepared"
    prepareDataset(planPath, dataRoot, output, progress=lambda message: None)
    return output


def refreshEvidence(output, paths):
    contract = readJson(output / "contract.json")
    for item in contract["artifactEvidence"]:
        if item["path"] in paths:
            path = output / item["path"]
            item.update(byteSize=path.stat().st_size, sha256=hashFile(path))
    contract.pop("contractId")
    contract["contractId"] = hashJson(contract)
    writeJson(output / "contract.json", contract)


def testPortableBundleVerifiesAfterRelocation(preparedSources):
    output = exportedFixture(preparedSources)
    relocated = output.parent / "relocated"
    assert output.resolve().parent == relocated.resolve().parent
    output.rename(relocated)
    result = verifyDataset(relocated, progress=lambda message: None)
    assert result["exportedImageCount"] == 10
    assert result["reservedTestRecordCount"] == 2
    assert main(["verifyDataset", "--datasetDir", str(relocated)]) == 0


def testChangedLabelCannotBeHiddenByUpdatingManifestHashes(preparedSources):
    output = exportedFixture(preparedSources)
    manifest = readJson(output / "manifest.json")
    row = manifest["samples"][0]
    label = output / row["labelPath"]
    content = label.read_text(encoding="utf-8")
    content = "1" + content[1:]
    label.write_text(content, encoding="utf-8", newline="\n")
    row["labelSha256"] = hashlib.sha256(label.read_bytes()).hexdigest()
    writeJson(output / "manifest.json", manifest)
    refreshEvidence(output, {"manifest.json", row["labelPath"]})
    with pytest.raises(ValueError, match="labels differ from native"):
        verifyDataset(output, progress=lambda message: None)


def testTestEntriesInDatasetYamlAreRejectedEvenWithMatchingHashes(preparedSources):
    output = exportedFixture(preparedSources)
    path = output / "unified25/dataset.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["test"] = "images/test"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    refreshEvidence(output, {"unified25/dataset.yaml"})
    with pytest.raises(ValueError, match="Dataset YAML"):
        verifyDataset(output, progress=lambda message: None)


def testUnexpectedFilesAndDamagedArtifactsFailVerification(preparedSources):
    output = exportedFixture(preparedSources)
    (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected files"):
        verifyDataset(output, progress=lambda message: None)
    with pytest.raises(SystemExit) as error:
        main(["verifyDataset", "--datasetDir", str(output)])
    assert error.value.code == 2
