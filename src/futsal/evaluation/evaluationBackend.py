from pathlib import Path
from types import SimpleNamespace

from futsal.datasets.datasetFiles import writeJson
from futsal.training.courtPose import poseClass, poseDatasetEntries
from futsal.training.trainingBackend import verifyLoader
from futsal.training.trainingConfig import courtRepresentation, stageParameters


def predictValidation(checkpoint, task, references, stage, trainingConfig, config, workDir, pointBoxes=None):
    """pointBoxes is required for courtPose, whose keypoints become native landmark detections."""
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.models.yolo.pose import PoseValidator
    from ultralytics.nn.tasks import load_checkpoint

    from futsal.evaluation.posePredictions import (
        bestInstance,
        landmarkDetections,
        landmarkMatches,
        poseInstances,
    )

    workDir = Path(workDir).resolve()
    root = workDir / "runtimeData" / task["taskId"]
    names = task["classes"]
    pose = courtRepresentation(trainingConfig) == "courtPose"
    if pose and pointBoxes is None:
        raise ValueError("courtPose evaluation requires the recipe point box fractions.")
    modelNames = [poseClass] if pose else names
    requested = trainingConfig.get("evaluationDevice", "cpu")
    device = torch.device("cpu" if requested == "cpu" else f"cuda:{requested}")
    model, _ = load_checkpoint(Path(checkpoint), device=device, fuse=False)
    if model.yaml.get("scale") != trainingConfig["modelSize"]:
        raise ValueError("Evaluation checkpoint differs from the recorded model size.")
    if model.names != dict(enumerate(modelNames)) or not model.end2end or model.model[-1].nc != len(modelNames):
        raise ValueError("Evaluation checkpoint differs from the selected stage ontology or YOLO26 mode.")
    if pose and list(model.kpt_shape) != [len(names), 3]:
        raise ValueError("Evaluation checkpoint differs from the court landmark keypoint shape.")
    configuredStage = next(item for item in trainingConfig["stages"] if item["taskId"] == task["taskId"])
    args = get_cfg(overrides={**stageParameters(trainingConfig, configuredStage), "mode": "val", "task": "pose" if pose else "detect",
                             "device": requested, "nms": False, "conf": config["confidenceFloor"],
                             "max_det": config["maxDetections"], "save_json": False,
                             "save_txt": False, "plots": False, "verbose": False,
                             "split": "val", "rect": True, "single_cls": False,
                             "data": str(root / "dataset.yaml"), "model": str(checkpoint)})
    model.args = args
    data = {"names": dict(enumerate(modelNames)), "nc": len(modelNames), "channels": 3,
            "val": str(root / "images" / "val"), "path": str(root), **(poseDatasetEntries(names) if pose else {})}
    from futsal.training.sharedRuntime import buildRuntimeDataset

    dataset = buildRuntimeDataset(args, workDir, task["taskId"], data, args.batch * 2, "val", int(model.stride.max()))
    taskRows = {"validation": task["rows"]}
    writeJson(workDir / f"{task['taskId']}LoaderAudit.json", verifyLoader(dataset, taskRows, "validation", workDir))
    loader = build_dataloader(dataset, batch=args.batch * 2, workers=0, shuffle=False,
                              rank=-1, device=device)
    expected = {(workDir / "runtimeData" / row["imagePath"]).resolve(): row for row in references}
    byRecord = {row["recordId"]: row for row in references}
    predictions = []

    class CapturingValidator(PoseValidator if pose else DetectionValidator):
        def run_callbacks(self, event):
            pass

        def update_metrics(self, preds, batch):
            for index, pred in enumerate(preds):
                prepared = self._prepare_batch(index, batch)
                reference = (byRecord[batch["recordId"][index]] if "recordId" in batch else
                             expected[Path(prepared["im_file"]).resolve()])
                scaled = self.scale_preds(self._prepare_pred(pred), prepared)
                row = {"recordId": reference["recordId"], "dimensions": reference["dimensions"],
                       "nativeOrder": self.seen + index}
                if pose:
                    instances = poseInstances(scaled)
                    row["detections"] = landmarkDetections(instances, reference["dimensions"], names, pointBoxes,
                                                           config["confidenceFloor"])
                    row["correctAtIou"] = landmarkMatches(self, row["detections"], reference)
                    row["poseInstance"] = bestInstance(instances)
                else:
                    row["detections"] = [{"classId": int(classId), "className": names[int(classId)],
                                          "confidence": float(confidence), "box": box.tolist()}
                                         for box, confidence, classId in zip(scaled["bboxes"].cpu(),
                                                                            scaled["conf"].cpu(),
                                                                            scaled["cls"].cpu(), strict=True)]
                    row["correctAtIou"] = self._process_batch(self._prepare_pred(pred), prepared)["tp"].tolist()
                predictions.append(row)
            super().update_metrics(preds, batch)

    validator = CapturingValidator(loader, save_dir=workDir / "nativeValidation" / task["taskId"],
                                   args=vars(args))
    context = SimpleNamespace(ema=SimpleNamespace(ema=model), device=device, data=data,
                              amp=False, loss_items={key: torch.tensor(0.0, device=device) for key in stage["results"]["losses"]},
                              stopper=SimpleNamespace(possible_stop=False), epoch=0, epochs=1,
                              world_size=1, label_loss_items=lambda loss, prefix: {})
    validator(trainer=context)
    writeJson(workDir / f"{task['taskId']}Arguments.json", vars(args))
    return predictions, {key: float(value) for key, value in validator.metrics.results_dict.items()}
