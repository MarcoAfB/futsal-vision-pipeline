from pathlib import Path

from futsal.datasets.datasetFiles import hashJson, readJson
from futsal.datasets.mediaStore import readBindings, resolveAsset
from futsal.datasets.sharedViews import verifyView
from futsal.evaluation.evaluationData import buildReferences
from futsal.inventory.catalog import hashFile


def selectTrackingData(datasetDir, mediaStoreDir):
    root, store = Path(datasetDir).resolve(), Path(mediaStoreDir).resolve()
    verifyView(root)
    plan, manifest = readJson(root / "plan.json"), readJson(root / "manifest.json")
    task = next((task for task in plan["recipe"]["tasks"] if task["taskId"] == "ballOnly"), None)
    if task is None or task["classes"] != ["ball"]:
        raise ValueError("Tracking requires a real ballOnly shared task view.")
    rows = [row for row in manifest["samples"] if row["taskId"] == "ballOnly" and row["split"] == "validation"]
    if not rows or any(row["origin"] != "own" for row in rows):
        raise ValueError("Tracking requires nonempty own validation data.")
    samples = {sample["recordId"]: sample for sample in plan["samples"]}
    intervals = {item["intervalId"]: item for video in plan["catalog"]["data"]["ownVideos"]
                 for item in video["reviewedIntervals"]}
    rows.sort(key=lambda row: (row["matchId"], row["sourceId"],
                               samples[row["recordId"]]["intervalId"], row["frameIndex"]))
    references = buildReferences(plan, {**task, "rows": rows})
    bindings = readBindings(store)
    images, segments = [], []
    previous = None
    for row, reference in zip(rows, references, strict=True):
        sample = samples[row["recordId"]]
        interval = intervals[sample["intervalId"]]
        if not interval["startFrame"] <= row["frameIndex"] <= interval["endFrame"] or len(reference["targets"]) > 1:
            raise ValueError("Ball tracking reference exceeds reviewed bounds or has multiple balls.")
        key = (row["matchId"], row["sourceId"], sample["intervalId"])
        if previous is None or key != previous[0] or row["frameIndex"] != previous[1] + 1:
            segments.append({"segmentId": f"{row['sourceId']}:{sample['intervalId']}:{row['frameIndex']}",
                             "matchId": row["matchId"], "sourceId": row["sourceId"],
                             "intervalId": sample["intervalId"], "startFrame": row["frameIndex"],
                             "endFrame": row["frameIndex"], "frameCount": 0,
                             "internalCutsStatus": interval["internalCutsStatus"]})
        segments[-1]["endFrame"] = row["frameIndex"]
        segments[-1]["frameCount"] += 1
        reference["segmentId"] = segments[-1]["segmentId"]
        image = resolveAsset(store, row["imageSha256"], bindings=bindings)
        if hashFile(image) != row["imageSha256"]:
            raise ValueError(f"Validation image integrity mismatch: {row['recordId']}.")
        images.append(str(image))
        previous = key, row["frameIndex"]
    excluded = [{"recordId": s["recordId"], "matchId": s["matchId"], "frameIndex": s["frameIndex"],
                 "reasons": s["tasks"]["ballOnly"]["reasons"]} for s in plan["samples"]
                if s["origin"] == "own" and plan["frozenSplit"]["assignments"].get(s["matchId"]) == "validation"
                and not s["tasks"]["ballOnly"]["eligible"]]
    selection = {"references": references, "segments": segments, "excluded": excluded,
                 "datasetContractSha256": hashFile(root / "contract.json"),
                 "metadataEvidence": [{"path": name, "sha256": hashFile(root / name)}
                                      for name in ("plan.json", "manifest.json")],
                 "frameCount": len(references), "positiveFrames": sum(bool(r["targets"]) for r in references),
                 "negativeFrames": sum(not r["targets"] for r in references), "testEvaluated": False}
    selection["selectionId"] = hashJson(selection)
    return selection, images
