import copy
from xml.etree import ElementTree

import cv2
import numpy as np
import pytest

from futsal.cli import main
from futsal.datasetExport import prepareDataset
from futsal.datasetFiles import readJson, verifyEvidence
from futsal.datasetPlan import buildDatasetPlan, writeDatasetPlan
from futsal.frameExtraction import readSelectedFrames


def testExportPreservesFramesPublicBytesAndReservedTest(preparedSources):
    dataRoot, configPath, planDir = preparedSources
    plan = buildDatasetPlan(configPath, dataRoot)
    planPath = writeDatasetPlan(plan, planDir, dataRoot)
    output = planDir.parent / "prepared"
    contract = prepareDataset(planPath, dataRoot, output, progress=lambda message: None)
    manifest = readJson(output / "manifest.json")
    assert contract["testExported"] is False
    assert manifest["summary"]["unified25"]["training"]["imageCount"] == 2
    assert manifest["summary"]["unified25"]["validation"]["imageCount"] == 2
    testIds = {sample["recordId"] for sample in plan["samples"] if sample["origin"] == "own"
               and plan["frozenSplit"]["assignments"][sample["matchId"]] == "test"}
    assert not testIds & {row["recordId"] for row in manifest["samples"]}
    for row in manifest["samples"]:
        if row["origin"] == "public":
            assert (output / row["imagePath"]).read_bytes() == (dataRoot / row["mediaPath"]).read_bytes()
        else:
            capture = cv2.VideoCapture(str(dataRoot / row["mediaPath"]))
            for _ in range(row["frameIndex"] + 1):
                success, expected = capture.read()
                assert success
            capture.release()
            assert np.array_equal(cv2.imread(str(output / row["imagePath"])), expected)
    assert not any("/test/" in item["path"] for item in contract["artifactEvidence"])
    verifyEvidence(dataRoot, plan["sourceEvidence"])


def testQuarantinePreviewIsLimitedToDevelopmentData(preparedSources):
    dataRoot, configPath, planDir = preparedSources
    path = dataRoot / "video1.xml"
    tree = ElementTree.parse(path)
    track = next(track for track in tree.getroot().findall("track") if track.find("points") is not None)
    track.append(copy.deepcopy(track.find("points")))
    tree.write(path, encoding="utf-8")
    plan = buildDatasetPlan(configPath, dataRoot)
    planPath = writeDatasetPlan(plan, planDir, dataRoot)
    output = planDir.parent / "prepared"
    prepareDataset(planPath, dataRoot, output, progress=lambda message: None)
    manifest = readJson(output / "manifest.json")
    assert len(manifest["reviews"]) == 1
    review = manifest["reviews"][0]
    assert review["recordId"] == "video1:0"
    assert set(review["quarantinedTasks"]) == {"unified25"}
    assert (output / review["imagePath"]).is_file()
    assert manifest["summary"]["unified25"]["validation"]["imageCount"] == 1
    assert manifest["summary"]["genericObjects"]["validation"]["imageCount"] == 2


def testChangedSourceOrDecoderFailureCannotPublishACompleteDataset(preparedSources, monkeypatch):
    dataRoot, configPath, planDir = preparedSources
    plan = buildDatasetPlan(configPath, dataRoot)
    planPath = writeDatasetPlan(plan, planDir, dataRoot)
    output = planDir.parent / "prepared"

    def brokenDecoder(*args):
        raise OSError("interrupted decoder")

    monkeypatch.setattr("futsal.datasetExport.readSelectedFrames", brokenDecoder)
    with pytest.raises(OSError, match="interrupted"):
        prepareDataset(planPath, dataRoot, output, progress=lambda message: None)
    assert not output.exists()
    assert not list(output.parent.glob(".prepared.*"))
    (dataRoot / "futsal4400/image.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Input changed"):
        prepareDataset(planPath, dataRoot, output, progress=lambda message: None)
    assert not output.exists()


def testSequentialExtractionDoesNotShiftRequestedFrameIndices(preparedSources):
    dataRoot, _, _ = preparedSources
    frames = list(readSelectedFrames(dataRoot / "video0.avi", [1]))
    assert [index for index, _ in frames] == [1]
    assert float(frames[0][1].mean()) == pytest.approx(40, abs=2)
    with pytest.raises(OSError, match="ended before"):
        list(readSelectedFrames(dataRoot / "video0.avi", [2]))


def testPrepareCommandPublishesOnceAndRefusesOverwrite(preparedSources):
    dataRoot, configPath, planDir = preparedSources
    planPath = writeDatasetPlan(buildDatasetPlan(configPath, dataRoot), planDir, dataRoot)
    output = planDir.parent / "prepared"
    arguments = ["prepareDataset", "--dataRoot", str(dataRoot), "--plan", str(planPath),
                 "--outputDir", str(output)]
    assert main(arguments) == 0
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
