import copy
from pathlib import Path

import pytest
import yaml

from futsal.datasets.datasetFiles import readJson
from futsal.datasets.mediaRegistration import registerMedia
from futsal.datasets.sharedViews import prepareView
from futsal.inventory.catalog import hashFile
from futsal.training.sharedRuntime import buildRuntimeDataset, prepareSharedRuntime
from futsal.training.trainingBackend import initializeBackend, verifyLoader
from futsal.training.trainingData import selectTrainingData


@pytest.mark.trainingIntegration
def testSharedReaderRejectsIncompleteJpegWithoutRepair(tmp_path):
    import cv2
    import numpy as np

    initializeBackend(tmp_path / "backend", 2)
    from ultralytics.cfg import get_cfg

    from futsal.training.sharedDataset import SharedDetectionDataset

    path = tmp_path / "incomplete.jpg"
    assert cv2.imwrite(str(path), np.full((32, 48, 3), 100, dtype=np.uint8))
    path.write_bytes(path.read_bytes()[:-2])
    original = hashFile(path)
    row = {"recordId": "source:0", "imagePath": "media/image", "imageSha256": original,
           "resolvedImage": str(path), "dimensions": [48, 32], "targets": []}
    dataset = SharedDetectionDataset([row], data={"names": {0: "player"}},
                                     hyp=get_cfg(), imgsz=64, augment=False, batch_size=1)
    with pytest.raises(ValueError, match="automatic repair is forbidden"):
        dataset[0]
    assert hashFile(path) == original
    assert not path.with_suffix(".npy").exists()


@pytest.mark.trainingIntegration
def testNativeSharedAndCopiedInputsMatchAcrossEveryTask(trainingBundle, preparedSources, tmp_path):
    from futsal.annotations.detectionGeometry import renderYoloLabels
    from futsal.datasets.datasetExport import sampleToken
    from futsal.datasets.mediaStore import resolveAsset
    from futsal.training.trainingData import copyRuntimeData

    bundle, _, config = trainingBundle
    store = tmp_path / "store"
    registerMedia(store, bundle)
    curriculum = tmp_path / "curriculum"
    prepareView(store, preparedSources[1], curriculum)
    recipe = yaml.safe_load(Path("configs/sharedDetection.yaml").read_text())
    recipe.pop("splitPath")
    recipePath = preparedSources[1].with_name("shared.yaml")
    recipePath.write_text(yaml.safe_dump(recipe))
    own = tmp_path / "own"
    prepareView(store, recipePath, own)
    initializeBackend(tmp_path / "backend", 2)
    import numpy as np
    import torch
    from ultralytics.cfg import get_cfg

    args = get_cfg(overrides={**config["parameters"], "imgsz": 64, "cache": False})
    for view in (curriculum, own):
        manifest = readJson(view / "manifest.json")
        for task in readJson(view / "plan.json")["recipe"]["tasks"]:
            taskId = task["taskId"]
            rows = [row for row in manifest["samples"] if row["taskId"] == taskId and row["split"] == "validation"]
            if not rows:
                continue
            selected = {"tasks": {taskId: {"classes": task["classes"], "training": [], "validation": rows}}}
            run = tmp_path / f"{view.name}{taskId}Shared"
            run.mkdir()
            prepareSharedRuntime(store, run, selected)
            copied = copy.deepcopy(selected)
            sourceRoot = tmp_path / f"{view.name}{taskId}Source"
            for row in copied["tasks"][taskId]["validation"]:
                source = resolveAsset(store, row["imageSha256"])
                row["imagePath"] = f"{taskId}/images/val/{sampleToken(row)}{source.suffix}"
                row["labelPath"] = f"{taskId}/labels/val/{sampleToken(row)}.txt"
                for key, content in (("imagePath", source.read_bytes()), ("labelPath", renderYoloLabels(row["targets"]).encode())):
                    path = sourceRoot / row[key]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
            copiedRun = tmp_path / f"{view.name}{taskId}Copied"
            copiedRun.mkdir()
            copyRuntimeData(sourceRoot, copiedRun, copied)
            datasets = []
            for current in (run, copiedRun):
                data = {"names": dict(enumerate(task["classes"])), "nc": len(task["classes"]), "channels": 3,
                        "val": str(current / "runtimeData" / taskId / "images/val")}
                datasets.append(buildRuntimeDataset(args, current, taskId, data, 2, "val", 32))
            first, second = datasets
            assert len(first) == len(second)
            for index in range(len(first)):
                assert np.array_equal(first.labels[index]["cls"], second.labels[index]["cls"])
                assert np.array_equal(first.labels[index]["bboxes"], second.labels[index]["bboxes"])
                for key in ("img", "bboxes", "cls"):
                    assert torch.equal(first[index][key], second[index][key])


@pytest.mark.trainingIntegration
def testNativeSharedReaderRetainsLogicalRecordsAndIgnoresAdjacentCaches(trainingBundle, preparedSources, tmp_path):
    bundle, _, config = trainingBundle
    store, view = tmp_path / "store", tmp_path / "view"
    registerMedia(store, bundle)
    prepareView(store, preparedSources[1], view)
    selection = selectTrainingData(view, config)
    task = selection["tasks"]["genericObjects"]
    second = copy.deepcopy(task["training"][0])
    second["recordId"] += ":second"
    second["targets"] = []
    task["training"].append(second)
    run = tmp_path / "run"
    run.mkdir()
    prepareSharedRuntime(store, run, selection)
    initializeBackend(tmp_path / "backend", 2)
    import numpy as np
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader

    args = get_cfg(overrides={**config["parameters"], "imgsz": 64, "cache": False})
    data = {"names": dict(enumerate(task["classes"])), "nc": 2, "channels": 3}
    snapshot = readJson(run / "runtimeData/genericObjects/sharedSelection.json")
    source = Path(snapshot["training"][0]["resolvedImage"])
    cache = source.with_suffix(".npy")
    np.save(cache, np.full((1, 1, 3), 255, dtype=np.uint8))
    before = {path: hashFile(path) for path in source.parent.iterdir() if path.is_file()}
    dataset = buildRuntimeDataset(args, run, "genericObjects", data, 2, "train", 32)
    assert verifyLoader(dataset, task, "training", run)["imageCount"] == 2
    loader = build_dataloader(dataset, batch=2, workers=0, shuffle=False, rank=-1)
    batch = next(iter(loader))
    assert set(batch["recordId"]) == {row["recordId"] for row in task["training"]}
    assert len(set(batch["assetId"])) == 1
    assert batch["img"].shape[-2:] == (64, 64)
    assert len(batch["cls"]) == len(task["training"][0]["targets"])
    assert before == {path: hashFile(path) for path in before}
    assert set(before) == {path for path in source.parent.iterdir() if path.is_file()}


@pytest.mark.trainingIntegration
def testNativeSharedCourtTrainsAndEvaluatesRealAndMixedWithoutCopies(projectionBundle, syntheticSource, tmp_path, monkeypatch):
    from futsal.evaluation.courtComparison import compareCourtTraining
    from futsal.evaluation.evaluationRun import evaluateRun
    from futsal.training.modelWeights import prepareWeights
    from futsal.training.trainingRun import runTraining

    parent, supplement, recipe, _ = projectionBundle
    store, view = tmp_path / "store", tmp_path / "view"
    registerMedia(store, parent)
    registerMedia(store, supplement)
    prepareView(store, recipe, view)
    initializeBackend(tmp_path / "setup", 2)
    import torch
    from ultralytics.nn.tasks import DetectionModel

    def rejectNetwork(*args, **kwargs):
        raise AssertionError("Native shared fixtures must remain offline.")

    monkeypatch.setattr("requests.sessions.Session.request", rejectNetwork)
    monkeypatch.setattr("urllib.request.urlopen", rejectNetwork)
    monkeypatch.setattr("futsal.training.modelWeights.downloadWeights", lambda url, path: torch.save(
        {"model": DetectionModel("yolo26n.yaml", nc=80, verbose=False)}, path))
    prepareWeights("n", tmp_path / "weights")
    config = yaml.safe_load(Path("configs/training/courtPoints.yaml").read_text())
    config.update(device="cpu", evaluationDevice="cpu")
    config["parameters"].update(imgsz=64, batch=1, nbs=2, workers=0, warmup_epochs=0.0, patience=0)
    config["stages"][0]["epochs"] = 1
    profile = tmp_path / "court.yaml"
    profile.write_text(yaml.safe_dump(config))
    evaluations = []
    for mixed in (False, True):
        runDir = tmp_path / f"run{mixed}"
        run = runTraining(view, profile, tmp_path / "weights/yolo26n.pt", runDir,
                          mediaStoreDir=store, syntheticDir=syntheticSource if mixed else None)
        assert run["status"] == "completed", run.get("error")
        assert run["runtimeIntegrityAfterRun"] and run["bundleIntegrityAfterRun"]["hashesVerified"]
        assert not list((runDir / "runtimeData").rglob("*.png"))
        stage = run["stages"][0]
        assert stage["results"]["checkpoints"]["best"]["classCount"] == 20
        assert readJson(runDir / stage["directory"] / "optimizer.json")["initialStateEntries"] == 0
        output = tmp_path / f"evaluation{mixed}"
        evaluation = evaluateRun(runDir, view, "configs/evaluation/researchValidation.yaml", "runValidation", output,
                                  workRoot=tmp_path / "work", mediaStoreDir=store)
        assert evaluation["status"] == "completed", evaluation.get("error")
        assert not list((Path(evaluation["workDirectory"]) / "runtimeData").rglob("*.png"))
        evaluations.append(output)
    result = compareCourtTraining(*evaluations, tmp_path / "comparison")
    assert result["status"] == "completed"
