import copy

import pytest

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.tracking.trackingData import selectTrackingData


def testSharedValidationSelectionRetainsNativeBoxesAndSeals(trackingView):
    view, store = trackingView
    selection, images = selectTrackingData(view, store)
    assert len(images) == selection["frameCount"] == 2
    assert selection["positiveFrames"] == 2 and selection["negativeFrames"] == 0
    assert len(selection["segments"]) == 1 and selection["testEvaluated"] is False
    reference = selection["references"][0]
    assert reference["targets"][0]["box"] == pytest.approx([8, 8, 12, 12])
    assignments = readJson(view / "plan.json")["frozenSplit"]["assignments"]
    assert assignments[reference["matchId"]] == "validation"
    manifest = view / "manifest.json"
    value = readJson(manifest)
    value["samples"][0]["split"] = "test"
    writeJson(manifest, value)
    with pytest.raises(ValueError):
        selectTrackingData(view, store)


def testGapsAndIntervalBoundariesResetWithoutDroppingShortOrNegativeFrames(trackingView, monkeypatch):
    view, store = trackingView
    plan, manifest = readJson(view / "plan.json"), readJson(view / "manifest.json")
    rows = [r for r in manifest["samples"] if r["split"] == "validation" and r["taskId"] == "ballOnly"]
    originals = {s["recordId"]: s for s in plan["samples"]}
    row = copy.deepcopy(rows[-1])
    row.update(recordId="extra:3", frameIndex=3, targets=[])
    native = copy.deepcopy(originals[rows[-1]["recordId"]])
    native.update(recordId=row["recordId"], frameIndex=3)
    intervalId = native["intervalId"]
    for video in plan["catalog"]["data"]["ownVideos"]:
        for interval in video["reviewedIntervals"]:
            if interval["intervalId"] == intervalId:
                interval["endFrame"] = 3
    plan["samples"].append(native)
    manifest["samples"].append(row)
    writeJson(view / "plan.json", plan)
    writeJson(view / "manifest.json", manifest)
    monkeypatch.setattr("futsal.tracking.trackingData.verifyView", lambda root: None)
    selection, _ = selectTrackingData(view, store)
    assert [s["frameCount"] for s in selection["segments"]] == [2, 1]
    assert selection["negativeFrames"] == 1
