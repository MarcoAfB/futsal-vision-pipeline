import os
import uuid
from collections import defaultdict
from pathlib import Path

import cv2

from futsal.datasets.datasetExport import encodeImage
from futsal.datasets.datasetFiles import readJson, verifyEvidence
from futsal.datasets.frameExtraction import readSelectedFrames
from futsal.datasets.mediaRegistration import realCollection
from futsal.datasets.mediaStore import (
    bindAsset,
    publishCollection,
    readBindings,
    storeCollections,
    storeLock,
)
from futsal.inventory.catalog import hashFile, resolveSourcePath


def extractSharedImages(storeDir, plan, missing, dataRoot, progress):
    dataRoot = Path(dataRoot).resolve()
    aliases = {}
    if (dataRoot / "sourceManifest.json").is_file():
        manifest = readJson(dataRoot / "sourceManifest.json")
        aliases = manifest.get("sourceAliases", {})
    samples = [s for s in plan["samples"] if s["recordId"] in missing]
    if any(s["origin"] == "own" and plan["frozenSplit"]["assignments"][s["matchId"]] == "test" for s in samples):
        raise ValueError("Shared extraction refuses reserved test images.")
    paths = {s[key] for s in samples for key in ("mediaPath", "annotationPath")}
    evidence = [{**item, "path": aliases.get(item["path"], item["path"])}
                for item in plan["sourceEvidence"] if item["path"] in paths]
    if len(evidence) != len(paths):
        raise ValueError("Missing native extraction source evidence.")
    verifyEvidence(dataRoot, evidence)
    images = {}
    with storeLock(storeDir) as root:
        for collection in storeCollections(root):
            if collection["sourcePolicy"] == "realOnly":
                images.update({key: item for key, item in collection["images"].items() if key in missing})
        bindings = readBindings(root)
        directory = root / "assets"
        directory.mkdir(exist_ok=True)

        def publish(sample, content):
            import hashlib

            assetId = hashlib.sha256(content).hexdigest()
            path = directory / f"{assetId}.png"
            if not path.exists():
                temporary = directory / f".{uuid.uuid4().hex}.partial"
                try:
                    temporary.write_bytes(content)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            bindAsset(bindings, assetId, path)
            images[sample["recordId"]] = {"assetId": assetId, "byteSize": len(content)}

        videos = defaultdict(dict)
        for sample in samples:
            if sample["recordId"] in images:
                continue
            if sample["origin"] == "public":
                source = resolveSourcePath(dataRoot, aliases.get(sample["mediaPath"], sample["mediaPath"]))
                assetId = hashFile(source)
                bindAsset(bindings, assetId, source)
                images[sample["recordId"]] = {"assetId": assetId, "byteSize": source.stat().st_size}
            else:
                videos[sample["mediaPath"]][sample["frameIndex"]] = sample
        for video, byIndex in sorted(videos.items()):
            progress(f"Extracting missing shared development frames once: {video} ({len(byIndex)})")
            for index, image in readSelectedFrames(resolveSourcePath(dataRoot, aliases.get(video, video)), byIndex):
                sample = byIndex[index]
                if [image.shape[1], image.shape[0]] != sample["dimensions"]:
                    raise ValueError("Shared frame dimensions differ from native annotations.")
                publish(sample, encodeImage(image, ".png", [cv2.IMWRITE_PNG_COMPRESSION, 1]))
        if set(images) != set(missing):
            raise ValueError("Shared extraction did not provide every requested record.")
        verifyEvidence(dataRoot, evidence)
        collection = realCollection(plan, images, None)
        collection["purpose"] = "extractedAssets"
        publishCollection(root, collection, bindings)
    return images
