import json
import logging
import math
import os
import time
from pathlib import Path

from futsal.datasets.datasetFiles import verifyEvidence, writeJson
from futsal.inventory.catalog import hashFile, readYaml
from futsal.training.checkpointTransfer import classificationKeys, loadAudited
from futsal.training.memoryRecovery import requireFixedBatch
from futsal.training.trainingConfig import stageParameters


def initializeBackend(runtimeDir, numThreads):
    runtimeDir = Path(runtimeDir).resolve()
    for name, directory in (("YOLO_CONFIG_DIR", "settings"), ("MPLCONFIGDIR", "matplotlib"),
                            ("TORCH_HOME", "torch")):
        path = runtimeDir / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from ultralytics import settings

    torch.set_num_threads(numThreads)
    settings.file = runtimeDir / "settings" / "Ultralytics" / "settings.json"
    settings.file_path = settings.file
    settings.update({key: False for key in ("sync", "clearml", "comet", "dvc", "mlflow",
                                           "raytune", "tensorboard", "wandb", "vscode_msg",
                                           "openvino_msg")})
    settings.update({"datasets_dir": str(runtimeDir / "datasets"),
                     "weights_dir": str(runtimeDir / "weights"),
                     "runs_dir": str(runtimeDir / "runs")})


def verifyLoader(dataset, task, split, runDir):
    import numpy as np

    from futsal.annotations.detectionGeometry import renderYoloLabels

    if hasattr(dataset, "sharedRows"):
        expected = {row["recordId"]: row for row in task[split]}
        actual = {label["recordId"]: label for label in dataset.labels}
    else:
        expected = {(Path(runDir) / "runtimeData" / row["imagePath"]).resolve(): row
                    for row in task[split]}
        actual = {Path(label["im_file"]).resolve(): label for label in dataset.labels}
    if set(actual) != set(expected) or len(dataset.labels) != len(expected):
        raise ValueError(f"Native loader changed {split} image membership.")
    for path, label in actual.items():
        row = expected[path]
        content = (renderYoloLabels(row["targets"]).splitlines() if hasattr(dataset, "sharedRows") else
                   (Path(runDir) / "runtimeData" / row["labelPath"]).read_text().splitlines())
        wanted = np.array(sorted(tuple(map(float, line.split())) for line in content)).reshape(-1, 5)
        observed = np.concatenate((label["cls"], label["bboxes"]), axis=1)
        observed = np.array(sorted(map(tuple, observed))).reshape(-1, 5)
        if wanted.shape != observed.shape or not np.allclose(wanted, observed, rtol=0, atol=1e-6):
            raise ValueError(f"Native loader discarded or changed annotations: {row['recordId']}.")
    return {"imageCount": len(actual), "annotationsVerified": True}


def validationMetrics(metrics, names, support, purpose="technicalSmoke"):
    indices = {int(classId): index for index, classId in enumerate(metrics.ap_class_index)}
    perClass = {}
    for classId, name in enumerate(names):
        values = metrics.class_result(indices[classId]) if classId in indices else (0, 0, 0, 0)
        perClass[name] = {"support": support[name], "supported": support[name] > 0,
                          **{key: float(value) if support[name] else None for key, value in
                             zip(("precision", "recall", "ap50", "ap50To95"), values, strict=True)}}
    aggregate = {key: float(value) for key, value in metrics.results_dict.items()}
    if any(not math.isfinite(value) for value in aggregate.values()) or any(
        value is not None and not math.isfinite(value) for item in perClass.values()
        for key, value in item.items() if key not in {"support", "supported"}
    ):
        raise ValueError("Validation returned nonfinite metrics.")
    return {"interpretation": "technicalSmokeOnly" if purpose == "technicalSmoke" else purpose,
            "aggregate": aggregate, "perClass": perClass}


def trainStage(config, task, stage, inputWeights, stageDir, runDir, runtimeEvidence, *, batchRecovery="native"):
    from ultralytics.utils import LOGGER

    logDir = Path(runDir) / "logs"
    logDir.mkdir(exist_ok=True)
    handler = logging.FileHandler(logDir / f"{stage['taskId']}.log", encoding="utf-8")
    LOGGER.addHandler(handler)
    try:
        return trainNativeStage(config, task, stage, inputWeights, stageDir, runDir, runtimeEvidence,
                                batchRecovery=batchRecovery)
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


def trainNativeStage(config, task, stage, inputWeights, stageDir, runDir, runtimeEvidence, *, batchRecovery="native"):
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.tasks import load_checkpoint
    from ultralytics.utils.torch_utils import strip_optimizer

    stageDir, runDir = Path(stageDir), Path(runDir)

    class AuditedDetectionTrainer(DetectionTrainer):
        def _build_train_pipeline(self):
            if batchRecovery == "fixed":
                requireFixedBatch(self.batch_size, parameters["batch"], getattr(self, "_oom_retries", 0))
            return super()._build_train_pipeline()

        def build_dataset(self, img_path, mode="train", batch=None):
            from ultralytics.utils.torch_utils import unwrap_model

            from futsal.training.sharedRuntime import buildRuntimeDataset

            stride = max(int(unwrap_model(self.model).stride.max()), 32)
            return buildRuntimeDataset(self.args, runDir, stage["taskId"], self.data, batch, mode, stride)

        def run_callbacks(self, event):
            for key, values in self.callbacks.items():
                self.callbacks[key] = [callback for callback in values if callback.__module__ in
                                       {"ultralytics.utils.callbacks.base", __name__}]
            super().run_callbacks(event)

        def get_dataset(self):
            data, _ = readYaml(Path(self.args.data))
            root = runDir / "runtimeData" / stage["taskId"]
            expected = {"path": str(root), "train": "images/train", "val": "images/val",
                        "names": dict(enumerate(task["classes"]))}
            if data != expected:
                raise ValueError("Runtime dataset configuration differs from the selected task.")
            return {**data, "train": str(root / data["train"]), "val": str(root / data["val"]),
                    "nc": len(task["classes"]), "channels": 3, "yaml_file": self.args.data}

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
            if weights is None or weights.yaml.get("scale") != config["modelSize"]:
                raise ValueError("Training requires the selected YOLO26 pretrained model scale.")
            if not weights.yaml.get("end2end") or weights.yaml.get("head", [])[-1][2] != "Detect":
                raise ValueError("Training requires an unfused YOLO26 detection checkpoint.")
            audit = loadAudited(model, weights)
            writeJson(stageDir / "transferAudit.json", audit)
            return model

        def final_eval(self):
            for path in (self.last, self.best):
                if path.is_file():
                    strip_optimizer(path)
            if not self.best.is_file():
                raise ValueError("Training did not produce a best checkpoint.")
            originalEma = self.ema.ema
            try:
                self.ema.ema, _ = load_checkpoint(self.best, device=self.device, fuse=False)
                self.ema.ema.args = self.args
                self.metrics = self.validator(trainer=self)
                self.metrics.pop("fitness", None)
            finally:
                self.ema.ema = originalEma

    parameters = stageParameters(config, stage)
    overrides = {**parameters, "model": str(Path(inputWeights).resolve()),
                 "data": str(runDir / "runtimeData" / stage["taskId"] / "dataset.yaml"),
                 "project": str(stageDir.parent), "name": stageDir.name, "exist_ok": False,
                 "epochs": stage["epochs"], "device": config["device"], "seed": config["seed"],
                 "task": "detect", "mode": "train", "resume": False, "pretrained": True,
                 "cls_remap": True, "split": "val", "val": True, "fraction": 1.0,
                 "single_cls": False, "classes": None, "compile": False,
                 "nbs": parameters.get("nbs", 2), "nms": False}
    trainer = AuditedDetectionTrainer(overrides=overrides)
    trainingCounts = {"batches": 0, "optimizerUpdates": 0}
    if trainer.save_dir.resolve() != stageDir.resolve():
        raise ValueError("Backend changed the requested new stage directory.")

    def beforeTraining(current):
        current.optimizer.register_step_post_hook(afterOptimizerStep)
        verifyEvidence(runDir, runtimeEvidence)
        loaders = {split: verifyLoader(loader.dataset, task, split, runDir) for split, loader in
                   (("training", current.train_loader), ("validation", current.test_loader))}
        writeJson(stageDir / "loaderAudit.json", loaders)
        writeJson(stageDir / "effectiveArguments.json", vars(current.args))
        writeJson(stageDir / "optimizer.json", {"type": type(current.optimizer).__name__,
                  "initialStateEntries": len(current.optimizer.state), "resume": bool(current.resume),
                  "accumulate": current.accumulate,
                  "paramGroups": [{key: value for key, value in group.items() if key != "params"}
                                  for group in current.optimizer.param_groups]})
        if current.optimizer.state or current.start_epoch != 0:
            raise ValueError("Each stage must start a fresh optimizer and epoch zero.")

    def afterBatch(current):
        trainingCounts["batches"] += 1
        losses = {key: float(value.detach().cpu()) for key, value in current.loss_items.items()}
        if not all(math.isfinite(value) for value in losses.values()):
            raise ValueError("Training returned nonfinite losses.")
        with (stageDir / "losses.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps({"epoch": current.epoch, "losses": losses}, allow_nan=False) + "\n")

    def afterEpoch(current):
        if not all(bool(torch.isfinite(value).all()) for model in (current.model, current.ema.ema)
                   for value in model.state_dict().values()):
            raise ValueError("Training produced nonfinite model tensors.")

    def afterOptimizerStep(optimizer, args, kwargs):
        trainingCounts["optimizerUpdates"] += 1

    history = []

    def afterFitEpoch(current):
        row = {"epoch": current.epoch + 1, "fitness": float(current.fitness),
               "bestFitness": float(current.best_fitness), "accumulate": current.accumulate,
               "learningRates": [group["lr"] for group in current.optimizer.param_groups],
               **trainingCounts}
        if not math.isfinite(row["fitness"]) or not math.isfinite(row["bestFitness"]):
            raise ValueError("Training returned nonfinite checkpoint selection fitness.")
        history.append(row)
        with (stageDir / "epochs.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, allow_nan=False) + "\n")

    trainer.add_callback("on_pretrain_routine_end", beforeTraining)
    trainer.add_callback("on_train_batch_end", afterBatch)
    trainer.add_callback("on_train_epoch_end", afterEpoch)
    trainer.add_callback("on_fit_epoch_end", afterFitEpoch)
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    started = time.perf_counter()
    trainer.train()
    writeJson(stageDir / "effectiveArguments.json", {**vars(trainer.args), "batch": trainer.batch_size})
    elapsed = time.perf_counter() - started
    verifyEvidence(runDir, runtimeEvidence)
    names = task["classes"]
    metrics = trainer.validator.metrics
    if trainer.validator.seen != len(task["validation"]) or any(
        int(metrics.nt_per_class[index]) != task["validationSupport"][name]
        for index, name in enumerate(names)
    ):
        raise ValueError("Validation did not process the exact selected annotations.")
    result = validationMetrics(metrics, names, task["validationSupport"], config["purpose"])
    result["training"] = {"requestedEpochs": stage["epochs"], "completedEpochs": len(history),
                          "requestedBatch": parameters["batch"], "effectiveBatch": trainer.batch_size,
                          "batchRecovery": batchRecovery,
                          **trainingCounts,
                          "bestEpoch": max(row["epoch"] for row in history if row["fitness"] == history[-1]["bestFitness"]),
                          "selectionCriterion": "nativeValidationMap50To95", "elapsedSeconds": elapsed,
                          "device": str(trainer.device), "amp": bool(trainer.amp),
                          "peakCudaAllocatedBytes": torch.cuda.max_memory_allocated(trainer.device)
                          if trainer.device.type == "cuda" else None,
                          "peakCudaReservedBytes": torch.cuda.max_memory_reserved(trainer.device)
                          if trainer.device.type == "cuda" else None}
    result["validationModel"] = "bestCheckpointReloadedAsFp32"
    checkpoints = {}
    for kind in ("best", "last"):
        path = Path(getattr(trainer, kind))
        model, _ = load_checkpoint(path, device="cpu", fuse=False)
        if model.names != dict(enumerate(names)):
            raise ValueError(f"{kind} checkpoint ontology differs from the stage classes.")
        if not model.end2end or model.model[-1].nc != len(names):
            raise ValueError(f"{kind} checkpoint lost the YOLO26 end-to-end detection contract.")
        classificationKeys(model)
        if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise ValueError(f"{kind} checkpoint contains nonfinite tensors.")
        checkpoints[kind] = {"path": path.relative_to(runDir).as_posix(),
                             "sha256": hashFile(path), "byteSize": path.stat().st_size,
                             "classCount": len(names), "names": names, "end2end": model.end2end}
    result["checkpoints"] = checkpoints
    result["losses"] = {key: float(value.detach().cpu()) for key, value in trainer.tloss.items()}
    writeJson(stageDir / "metrics.json", result)
    return result
