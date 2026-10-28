import shutil
import urllib.request
from pathlib import Path

from futsal.catalog import hashFile
from futsal.datasetFiles import readJson, stagedDirectory, writeJson


def downloadWeights(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "FutsalResearch/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def prepareWeights(modelSize, outputDir):
    if modelSize not in {"n", "s", "m", "l", "x"}:
        raise ValueError("YOLO26 modelSize must be n, s, m, l or x.")
    filename = f"yolo26{modelSize}.pt"
    url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{filename}"
    with stagedDirectory(outputDir, Path(__file__).resolve().parent) as staging:
        destination = staging / filename
        downloadWeights(url, destination)
        if destination.stat().st_size == 0:
            raise ValueError("The official weight download is empty.")
        metadata = {"schemaVersion": 1, "modelFamily": "yolo26", "modelSize": modelSize,
                    "filename": filename, "sourceUrl": url, "release": "v8.4.0",
                    "byteSize": destination.stat().st_size, "sha256": hashFile(destination)}
        writeJson(staging / "weights.json", metadata)
    return metadata


def verifyWeights(path, modelSize):
    path = Path(path).resolve()
    metadata = readJson(path.with_name("weights.json"))
    expectedUrl = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26{modelSize}.pt"
    if metadata.get("schemaVersion") != 1 or metadata.get("modelFamily") != "yolo26" or (
        metadata.get("modelSize") != modelSize or metadata.get("filename") != path.name
        or metadata.get("sourceUrl") != expectedUrl
    ):
        raise ValueError("Initial weight provenance does not match the requested official YOLO26 model.")
    if path.stat().st_size != metadata["byteSize"] or hashFile(path) != metadata["sha256"]:
        raise ValueError("Initial weights changed since preparation.")
    return metadata
