import pytest
import yaml

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.evaluation.evaluationData import writeJsonLines
from futsal.evaluation.trackingRun import trackRuns
from futsal.inventory.catalog import hashFile
from futsal.reporting.researchReport import renderReport, verifyEvaluation
from futsal.training.trainingRun import fileEvidence


@pytest.fixture
def trackedCampaign(trackingView, tmp_path, monkeypatch):
    from futsal.tracking.trackingConfig import loadTrackingConfig

    view, store = trackingView
    run = tmp_path / "labelledRun"
    run.mkdir()
    rows = [r for r in readJson(view / "manifest.json")["samples"] if r["taskId"] == "ballOnly"]
    task = {"classes": ["ball"], **{split: [r for r in rows if r["split"] == split] for split in ("training", "validation")}}
    writeJson(run / "selection.json", {"tasks": {"ballOnly": task}})
    (run / "best.pt").write_bytes(b"isolated software fixture; not a scientific checkpoint")
    writeJson(run / "run.json", {"status": "completed", "testEvaluated": False,
        "configuration": {"modelSize": "m", "seed": 42, "purpose": "scientificTraining", "sourcePolicy": "realOnly",
                          "parameters": {"batch": 4}, "stages": [{"taskId": "ballOnly", "epochs": 1}]},
        "selectionSha256": hashFile(run / "selection.json"),
        "stages": [{"taskId": "ballOnly", "status": "completed", "results": {"checkpoints": {
            "best": {"path": "best.pt", "sha256": hashFile(run / "best.pt")}}}}]})
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(yaml.safe_dump({**loadTrackingConfig("configs/evaluation/tracking.yaml"), "trackers": ["bytetrack"]}))
    monkeypatch.setattr("futsal.evaluation.trackingRun.preflightTracking", lambda *args: (
        {"packages": {"lap": "0.5.12"}, "requestedDevice": "cpu"}, {"bytetrack": "tracker_type: bytetrack\n"}))
    monkeypatch.setattr("futsal.evaluation.trackingRun.inspectTrackedModels", lambda *args: None)

    def snapshot(directory):
        code = directory / "code"
        code.mkdir()
        source = code / "fixture.py"
        source.write_text("fixture = True\n")
        return {"files": [fileEvidence(source, directory)]}

    calls = []

    def worker(requestPath):
        request = readJson(requestPath)
        assert request["classNames"] == ["ball"] and request["size"] == "m"
        calls.append(request["trackerPath"])
        predictions = [{**reference, "detections": [
            {"classId": 0, "className": "ball", "box": target["box"], "confidence": 0.9,
             "trackId": 1 if request["trackerPath"] else None} for target in reference["targets"]]}
            for reference in request["references"]]
        writeJsonLines(requestPath.parent / "predictions.jsonl", predictions)
        state = {"status": "completed", "frames": len(predictions)}
        writeJson(requestPath.parent / "worker.json", state)
        return state

    monkeypatch.setattr("futsal.evaluation.trackingRun.snapshotCode", snapshot)
    monkeypatch.setattr("futsal.reporting.trackingReport.archiveRenderer", lambda root: {"files": []})
    monkeypatch.setattr("futsal.evaluation.trackingRun.launchTrackingWorker", worker)
    return f"mRun={run}", view, store, protocol, calls


def testAnyLabelledRunIsTrackedSealedReplayedAndResumed(trackedCampaign, tmp_path):
    spec, view, store, protocol, calls = trackedCampaign
    root = tmp_path / "tracking"
    state = trackRuns([spec], view, store, root, configPath=protocol)
    assert state["status"] == "completed" and state["kind"] == "trackingComparison"
    assert [c["conditionId"] for c in state["conditions"]] == ["mRun_detector", "mRun_bytetrack"]
    assert verifyEvaluation(root)[0]["testEvaluated"] is False and len(calls) == 2
    assert state["selectionEvidence"]["taskId"] == "ballOnly" and state["selectionEvidence"]["classes"] == ["ball"]
    report = renderReport(root)
    assert "Tracking comparison" in report.read_text()
    summary = (root / "tables/summary.csv").read_text()
    assert "mRun,m,bytetrack,completed" in summary
    resumed = trackRuns([spec], view, store, tmp_path / "resumed", configPath=protocol, resumeFrom=root)
    assert resumed["status"] == "completed" and len(calls) == 2
    replay = renderReport(root, tmp_path / "replay")
    assert (root / "tables/summary.csv").read_bytes() == (replay.parent / "tables/summary.csv").read_bytes()


def testRunWithDifferentFinalTaskOrOntologyStopsTheCampaign(trackedCampaign, tmp_path):
    spec, view, store, protocol, _ = trackedCampaign
    run = tmp_path / "labelledRun"
    record = readJson(run / "run.json")
    record["stages"][0]["taskId"] = "playerOnly"
    writeJson(run / "run.json", record)
    state = trackRuns([spec], view, store, tmp_path / "wrongTask", configPath=protocol)
    assert state["status"] == "failed" and "ballOnly" in state["error"] and not state["conditions"]
