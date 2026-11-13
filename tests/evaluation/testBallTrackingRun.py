from pathlib import Path

import pytest
import yaml

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.evaluation.ballTrackingRun import compareBallTrackers
from futsal.evaluation.ballTrackingRun import launchTrackingWorker as nativeWorkerLauncher
from futsal.evaluation.evaluationData import writeJsonLines
from futsal.inventory.catalog import hashFile
from futsal.reporting.researchReport import renderReport, verifyEvaluation
from futsal.training.trainingRun import fileEvidence


@pytest.fixture
def trackingCampaign(trackingView, tmp_path, monkeypatch):
    from futsal.tracking.trackingConfig import loadTrackingConfig

    view, store = trackingView
    run = tmp_path / "trainingN"
    run.mkdir()
    rows = [r for r in readJson(view / "manifest.json")["samples"] if r["taskId"] == "ballOnly"]
    task = {"classes": ["ball"], **{split: [r for r in rows if r["split"] == split]
                                  for split in ("training", "validation")}}
    selection = {"tasks": {"ballOnly": task}}
    writeJson(run / "selection.json", selection)
    (run / "best.pt").write_bytes(b"isolated software fixture; not a scientific checkpoint")
    writeJson(run / "run.json", {"status": "completed", "testEvaluated": False,
        "configuration": {"modelSize": "n", "seed": 42, "purpose": "scientificTraining", "sourcePolicy": "realOnly", "parameters": {"batch": 8}},
        "selectionSha256": hashFile(run / "selection.json"),
        "stages": [{"taskId": "ballOnly", "status": "completed", "results": {"checkpoints": {
            "best": {"path": "best.pt", "sha256": hashFile(run / "best.pt")}}}}]})
    runs = tmp_path / "runs.yaml"
    runs.write_text(yaml.safe_dump({"schemaVersion": 1, "runs": {"n": str(run)}}))
    config = {**loadTrackingConfig("configs/evaluation/ballTracking.yaml"), "trackers": ["bytetrack"]}
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(yaml.safe_dump(config))
    monkeypatch.setattr("futsal.evaluation.ballTrackingRun.preflightTracking", lambda *args: (
        {"packages": {"lap": "0.5.12"}, "requestedDevice": "cpu"}, {"bytetrack": "tracker_type: bytetrack\n"}))
    monkeypatch.setattr("futsal.evaluation.ballTrackingRun.inspectTrackingModels", lambda *args: None)
    calls = []

    def snapshot(directory):
        code = directory / "code"
        code.mkdir()
        source = code / "fixture.py"
        source.write_text("fixture = True\n")
        return {"files": [fileEvidence(source, directory)]}

    monkeypatch.setattr("futsal.evaluation.ballTrackingRun.snapshotCode", snapshot)
    monkeypatch.setattr("futsal.reporting.ballTrackingReport.archiveRenderer", lambda root: {"files": []})

    def worker(requestPath):
        request = readJson(requestPath)
        calls.append(request["trackerPath"])
        predictions = []
        for reference in request["references"]:
            detections = [{"classId": 0, "className": "ball", "box": target["box"], "confidence": 0.9,
                           "trackId": 1 if request["trackerPath"] else None} for target in reference["targets"]]
            predictions.append({**reference, "detections": detections})
        writeJsonLines(requestPath.parent / "predictions.jsonl", predictions)
        state = {"status": "completed", "frames": len(predictions)}
        writeJson(requestPath.parent / "worker.json", state)
        return state

    monkeypatch.setattr("futsal.evaluation.ballTrackingRun.launchTrackingWorker", worker)
    return runs, view, store, protocol, calls, worker


def testCampaignSealsCommonCohortAndResumesWithoutNewInference(trackingCampaign, tmp_path, monkeypatch):
    runs, view, store, protocol, calls, _ = trackingCampaign
    root = tmp_path / "comparison"
    state = compareBallTrackers(runs, view, store, root, configPath=protocol)
    assert state["status"] == "completed" and len(state["conditions"]) == 2
    assert len(calls) == 2 and verifyEvaluation(root)[0]["testEvaluated"] is False
    original = hashFile(root / "evaluationContract.json")
    resumed = tmp_path / "resumed"
    result = compareBallTrackers(runs, view, store, resumed, configPath=protocol, resumeFrom=root)
    assert result["status"] == "completed" and len(calls) == 2
    assert all("reusedFromContractId" in c for c in result["conditions"])
    assert hashFile(root / "evaluationContract.json") == original
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from resultTransfer import verifyReport

    assert verifyReport(root) is None
    monkeypatch.setattr("futsal.tracking.trackingBackend.NativeBallTracker", lambda *args: pytest.fail("Unexpected model loading"))
    report = renderReport(root)
    replay = renderReport(root, tmp_path / "replay")
    assert "Ball tracking comparison" in report.read_text()
    assert (root / "tables/summary.csv").read_bytes() == (replay.parent / "tables/summary.csv").read_bytes()
    assert hashFile(root / "evaluationContract.json") == original
    evidence = root / "evidence/conditions/yolo26n_detector/predictions.jsonl"
    evidence.write_text("tampered\n")
    with pytest.raises(ValueError):
        renderReport(root, tmp_path / "tamperedReport")
    failed = compareBallTrackers(runs, view, store, tmp_path / "tamperedResume", configPath=protocol, resumeFrom=root)
    assert failed["status"] == "failed" and not failed["conditions"]


def testPartialCampaignPreservesFailuresAndRetriesOnlyFailedConditions(trackingCampaign, tmp_path, monkeypatch):
    runs, view, store, protocol, calls, worker = trackingCampaign

    def failingWorker(path):
        request = readJson(path)
        if request["trackerPath"]:
            (path.parent / "predictions.jsonl").write_text("partial evidence\n")
            (path.parent / "failure.txt").write_text("fixture failure\n")
            return {"status": "failed", "frames": 1, "error": "fixture failure"}
        return worker(path)

    monkeypatch.setattr("futsal.evaluation.ballTrackingRun.launchTrackingWorker", failingWorker)
    root = tmp_path / "partial"
    state = compareBallTrackers(runs, view, store, root, configPath=protocol)
    assert state["status"] == "partial" and state["conditions"][1]["frames"] == 1
    assert not (root / "evidence/conditions/yolo26n_bytetrack/metrics.json").exists()
    assert (root / "evidence/conditions/yolo26n_bytetrack/predictions.jsonl").read_text() == "partial evidence\n"
    monkeypatch.setattr("futsal.evaluation.ballTrackingRun.launchTrackingWorker", worker)
    state = compareBallTrackers(runs, view, store, tmp_path / "retry", configPath=protocol, resumeFrom=root)
    assert state["status"] == "completed" and len(calls) == 2
    changed = readJson(runs.parent / "trainingN/run.json")
    changed["configuration"]["parameters"]["batch"] = 2
    writeJson(runs.parent / "trainingN/run.json", changed)
    state = compareBallTrackers(runs, view, store, tmp_path / "different", configPath=protocol, resumeFrom=root)
    assert state["status"] == "failed" and "identical" in state["error"]


def testMissingDependenciesStopBeforeConditions(trackingCampaign, tmp_path, monkeypatch):
    runs, view, store, protocol, calls, _ = trackingCampaign

    def missing(*args):
        raise ValueError("missing lap")

    monkeypatch.setattr("futsal.evaluation.ballTrackingRun.preflightTracking", missing)
    state = compareBallTrackers(runs, view, store, tmp_path / "missing", configPath=protocol)
    assert state["status"] == "failed" and "missing lap" in state["error"]
    assert not state["conditions"] and not calls


@pytest.mark.parametrize("failure", ["cohort", "leak", "checkpoint", "size"])
def testInvalidModelProvenanceStopsTheWholeCampaign(trackingCampaign, tmp_path, failure):
    runs, view, store, protocol, calls, _ = trackingCampaign
    root = runs.parent / "trainingN"
    run = readJson(root / "run.json")
    if failure in {"cohort", "leak"}:
        selection = readJson(root / "selection.json")
        task = selection["tasks"]["ballOnly"]
        if failure == "cohort":
            task["validation"].pop()
        else:
            task["training"].append({**task["validation"][0], "split": "training"})
        writeJson(root / "selection.json", selection)
        run["selectionSha256"] = hashFile(root / "selection.json")
    elif failure == "checkpoint":
        (root / "best.pt").write_bytes(b"changed checkpoint")
    else:
        run["configuration"]["modelSize"] = "x"
    writeJson(root / "run.json", run)
    state = compareBallTrackers(runs, view, store, tmp_path / "invalid", configPath=protocol)
    assert state["status"] == "failed" and not state["conditions"] and not calls



@pytest.mark.trainingIntegration
def testNativeWorkersExecuteInSeparateProcesses(trackingCampaign, tmp_path, monkeypatch):
    from futsal.tracking.trackingBackend import preflightTracking
    from futsal.tracking.trackingModels import inspectTrackingModels
    from futsal.training.trainingBackend import initializeBackend

    runs, view, store, protocol, _, _ = trackingCampaign
    initializeBackend(tmp_path / "backend", 2)
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    runDir = runs.parent / "trainingN"
    model = YOLO("yolo26n.yaml")
    model.model = DetectionModel("yolo26n.yaml", nc=1, verbose=False)
    model.model.names = {0: "ball"}
    model.model.end2end = True
    model.save(runDir / "best.pt")
    metadata = readJson(runDir / "run.json")
    metadata["stages"][0]["results"]["checkpoints"]["best"]["sha256"] = hashFile(runDir / "best.pt")
    writeJson(runDir / "run.json", metadata)
    config = yaml.safe_load(protocol.read_text())
    config["imgsz"] = 96
    protocol.write_text(yaml.safe_dump(config))
    monkeypatch.setattr("futsal.evaluation.ballTrackingRun.launchTrackingWorker", nativeWorkerLauncher)
    monkeypatch.setattr("futsal.evaluation.ballTrackingRun.preflightTracking", preflightTracking)
    monkeypatch.setattr("futsal.evaluation.ballTrackingRun.inspectTrackingModels", inspectTrackingModels)
    root = tmp_path / "nativeCampaign"
    state = compareBallTrackers(runs, view, store, root, configPath=protocol)
    assert state["status"] == "completed", state
    assert all(condition["frames"] == 2 for condition in state["conditions"])
    assert not (root / "work").exists()
    for condition in state["conditions"]:
        evidence = root / condition["evidenceDirectory"]
        arguments = readJson(evidence / "nativeArguments.json")
        assert arguments["end2end"] is True and arguments["device"] == "cpu"
        assert arguments["half"] is False and arguments["parameterDtype"] == "torch.float32"
        assert arguments["arguments"]["seed"] == 42
