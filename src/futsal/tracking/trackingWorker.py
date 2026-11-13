import argparse
import traceback
from pathlib import Path

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.inventory.catalog import hashFile
from futsal.tracking.trackingBackend import NativeBallTracker
from futsal.training.trainingBackend import initializeBackend
from futsal.training.trainingRun import utcNow


def runWorker(requestPath):
    import json

    import cv2

    request = readJson(requestPath)
    root = Path(requestPath).parent
    state = {"status": "running", "startedAt": utcNow(), "frames": 0}
    try:
        initializeBackend(root / "runtime", request["config"]["numThreads"])
        checkpoint = Path(request["checkpoint"])
        if hashFile(checkpoint) != request["checkpointSha256"]:
            raise ValueError("Checkpoint integrity changed before worker execution.")
        runner = NativeBallTracker(checkpoint, request["size"], request["config"], request["device"],
                                   request["trackerPath"])
        with (root / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for reference, path in zip(request["references"], request["images"], strict=True):
                if hashFile(Path(path)) != reference["imageSha256"]:
                    raise ValueError(f"Image integrity changed: {reference['recordId']}.")
                image = cv2.imread(path)
                if image is None or [image.shape[1], image.shape[0]] != reference["dimensions"]:
                    raise ValueError(f"Cannot decode expected image: {reference['recordId']}.")
                detections = runner.predict(image, reference["segmentId"])
                row = {key: reference[key] for key in ("recordId", "frameIndex", "segmentId", "dimensions")}
                stream.write(json.dumps({**row, "detections": detections}, allow_nan=False) + "\n")
                stream.flush()
                state["frames"] += 1
        if hashFile(checkpoint) != request["checkpointSha256"]:
            raise ValueError("Checkpoint integrity changed during worker execution.")
        writeJson(root / "nativeArguments.json", {"arguments": vars(runner.model.predictor.args),
                  "inputSize": runner.model.predictor.imgsz, "device": str(runner.model.predictor.device),
                  "half": bool(runner.model.predictor.model.fp16),
                  "parameterDtype": str(next(runner.model.model.parameters()).dtype),
                  "end2end": runner.model.predictor.model.end2end})
        state["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        (root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        state["endedAt"] = utcNow()
        writeJson(root / "worker.json", state)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    raise SystemExit(runWorker(parser.parse_args().request))
