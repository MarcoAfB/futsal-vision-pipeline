from pathlib import Path
from types import SimpleNamespace

from futsal.datasets.datasetFiles import writeJson
from futsal.training.trainingBackend import verifyLoader


def selectGlobalDetections(components, maxDetections):
    combined = []
    for componentId, detections, classMap in components:
        for index, detection in enumerate(detections):
            classId = classMap[detection["classId"]]
            combined.append({**detection, "classId": classId, "componentId": componentId,
                             "componentPredictionIndex": index})
    return sorted(combined, key=lambda item: -item["confidence"])[:maxDetections]


def predictConfiguration(components, task, references, trainingConfig, config, workDir):
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.nn.tasks import load_checkpoint

    workDir = Path(workDir).resolve()
    root = workDir / "runtimeData" / task["taskId"]
    names = task["classes"]
    requested = trainingConfig["evaluationDevice"]
    device = torch.device("cpu" if requested == "cpu" else f"cuda:{requested}")
    models = []
    for component in components:
        model, _ = load_checkpoint(Path(component["checkpointPath"]), device=device, fuse=False)
        if model.names != dict(enumerate(component["classes"])) or not model.end2end or model.model[-1].nc != len(component["classes"]):
            raise ValueError("Component checkpoint ontology or YOLO26 mode changed.")
        model.eval().float()
        model.set_head_attr(max_det=config["maxDetections"], agnostic_nms=False)
        models.append((component, model))
    args = get_cfg(overrides={**trainingConfig["parameters"], "mode": "val", "task": "detect",
        "device": requested, "nms": False, "conf": config["confidenceFloor"],
        "max_det": config["maxDetections"], "save_json": False, "save_txt": False,
        "plots": False, "verbose": False, "split": "val", "rect": True, "single_cls": False,
        "data": str(root / "dataset.yaml")})
    data = {"names": dict(enumerate(names)), "nc": len(names), "channels": 3,
            "val": str(root / "images/val"), "path": str(root)}
    strides = {int(model.stride.max()) for _, model in models}
    if len(strides) != 1:
        raise ValueError("Joint native validation requires identical input strides.")
    dataset = build_yolo_dataset(args, data["val"], args.batch * 2, data, mode="val", rect=True, stride=strides.pop())
    writeJson(workDir / "configurationLoaderAudit.json", verifyLoader(dataset, {"validation": task["rows"]}, "validation", workDir))
    loader = build_dataloader(dataset, batch=args.batch * 2, workers=0, shuffle=False, rank=-1, device=device)
    validator = DetectionValidator(loader, save_dir=workDir / "nativeConfiguration", args=vars(args))
    validator.device, validator.data, validator.training = device, data, True
    validator.args.quantize = None
    validator.init_metrics(SimpleNamespace(names=dict(enumerate(names)), end2end=True))
    byPath = {(workDir / "runtimeData" / row["imagePath"]).resolve(): row for row in references}
    predictions = []
    with torch.inference_mode():
        for rawBatch in loader:
            batch = validator.preprocess(rawBatch)
            outputs = [(component, validator.postprocess(model(batch["img"]))) for component, model in models]
            mergedBatch = []
            for index in range(len(batch["im_file"])):
                inputs = []
                for component, native in outputs:
                    pred = native[index]
                    detections = [{"classId": int(classId), "confidence": float(confidence), "box": box}
                                  for classId, confidence, box in zip(pred["cls"].cpu().tolist(), pred["conf"].cpu().tolist(),
                                                                      pred["bboxes"].cpu().tolist(), strict=True)]
                    mapping = {i: names.index(name) for i, name in enumerate(component["classes"])}
                    inputs.append((component["taskId"], detections, mapping))
                selected = selectGlobalDetections(inputs, config["maxDetections"])
                native = {"bboxes": torch.tensor([item["box"] for item in selected], device=device).reshape(-1, 4),
                          "conf": torch.tensor([item["confidence"] for item in selected], device=device),
                          "cls": torch.tensor([item["classId"] for item in selected], device=device),
                          "extra": torch.empty((len(selected), 0), device=device)}
                mergedBatch.append(native)
                prepared = validator._prepare_batch(index, batch)
                mask = validator._process_batch(native, prepared)["tp"].tolist()
                scaled = validator.scale_preds(native, prepared)
                reference = byPath[Path(prepared["im_file"]).resolve()]
                detections = [{**item, "className": names[item["classId"]], "box": box}
                              for item, box in zip(selected, scaled["bboxes"].cpu().tolist(), strict=True)]
                predictions.append({"recordId": reference["recordId"], "dimensions": reference["dimensions"],
                                    "nativeOrder": len(predictions), "detections": detections, "correctAtIou": mask})
            validator.update_metrics(mergedBatch, batch)
    metrics = {key: float(value) for key, value in validator.get_stats().items()}
    if validator.seen != len(references):
        raise ValueError("Joint validator changed image membership.")
    writeJson(workDir / "configurationArguments.json", vars(args))
    return predictions, metrics
