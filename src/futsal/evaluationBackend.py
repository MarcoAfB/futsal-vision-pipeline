from pathlib import Path
from types import SimpleNamespace

from futsal.datasetFiles import writeJson
from futsal.trainingBackend import verifyLoader


def predictValidation(checkpoint, task, references, stage, trainingConfig, config, workDir):
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.nn.tasks import load_checkpoint

    workDir = Path(workDir).resolve()
    root = workDir / "runtimeData" / task["taskId"]
    names = task["classes"]
    model, _ = load_checkpoint(Path(checkpoint), device="cpu", fuse=False)
    if model.names != dict(enumerate(names)) or not model.end2end or model.model[-1].nc != len(names):
        raise ValueError("Evaluation checkpoint differs from the selected stage ontology or YOLO26 mode.")
    args = get_cfg(overrides={**trainingConfig["parameters"], "mode": "val", "task": "detect",
                             "device": "cpu", "nms": False, "conf": config["confidenceFloor"],
                             "max_det": config["maxDetections"], "save_json": False,
                             "save_txt": False, "plots": False, "verbose": False,
                             "split": "val", "rect": True, "single_cls": False,
                             "data": str(root / "dataset.yaml"), "model": str(checkpoint)})
    model.args = args
    data = {"names": dict(enumerate(names)), "nc": len(names), "channels": 3,
            "val": str(root / "images" / "val"), "path": str(root)}
    dataset = build_yolo_dataset(args, data["val"], args.batch * 2, data, mode="val", rect=True,
                                 stride=int(model.stride.max()))
    taskRows = {"validation": task["rows"]}
    writeJson(workDir / f"{task['taskId']}LoaderAudit.json", verifyLoader(dataset, taskRows, "validation", workDir))
    loader = build_dataloader(dataset, batch=args.batch * 2, workers=0, shuffle=False,
                              rank=-1, device=torch.device("cpu"))
    expected = {(workDir / "runtimeData" / row["imagePath"]).resolve(): row for row in references}
    predictions = []

    class CapturingValidator(DetectionValidator):
        def run_callbacks(self, event):
            pass

        def update_metrics(self, preds, batch):
            for index, pred in enumerate(preds):
                prepared = self._prepare_batch(index, batch)
                reference = expected[Path(prepared["im_file"]).resolve()]
                scaled = self.scale_preds(self._prepare_pred(pred), prepared)
                detections = [{"classId": int(classId), "className": names[int(classId)],
                               "confidence": float(confidence), "box": box.tolist()}
                              for box, confidence, classId in zip(scaled["bboxes"].cpu(), scaled["conf"].cpu(),
                                                                 scaled["cls"].cpu(), strict=True)]
                predictions.append({"recordId": reference["recordId"], "dimensions": reference["dimensions"],
                                    "detections": detections,
                                    "nativeOrder": self.seen + index,
                                    "correctAtIou": self._process_batch(self._prepare_pred(pred), prepared)["tp"].tolist()})
            super().update_metrics(preds, batch)

    validator = CapturingValidator(loader, save_dir=workDir / "nativeValidation" / task["taskId"],
                                   args=vars(args))
    context = SimpleNamespace(ema=SimpleNamespace(ema=model), device=torch.device("cpu"), data=data,
                              amp=False, loss_items={key: torch.tensor(0.0) for key in stage["results"]["losses"]},
                              stopper=SimpleNamespace(possible_stop=False), epoch=0, epochs=1,
                              world_size=1, label_loss_items=lambda loss, prefix: {})
    validator(trainer=context)
    writeJson(workDir / f"{task['taskId']}Arguments.json", vars(args))
    return predictions, {key: float(value) for key, value in validator.metrics.results_dict.items()}
