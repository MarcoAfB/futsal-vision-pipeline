import copy
from pathlib import Path

from futsal.datasets.datasetFiles import (
    hashJson,
    readJson,
    stagedDirectory,
    validateNewOutput,
    writeJson,
)
from futsal.inventory.catalog import (
    relativePathValue,
    resolveSourcePath,
    sourceAliases,
    validateSources,
)
from futsal.inventory.yoloInventory import scanFiles
from futsal.training.trainingData import requireDiskSpace


def nativeCatalog(data):
    result = copy.deepcopy(data)
    result.update(schemaVersion=2, reviewPolicy="catalogReviewedIntervals", syntheticSources=[])
    result.pop("legacyReviewPath", None)
    for video in result["ownVideos"]:
        video.pop("legacyMatchId", None)
        for field, folder in (("videoPath", "videos"), ("annotationPath", "annotations")):
            video[field] = f"own/{folder}/{Path(video[field]).name}"
        for interval in video["reviewedIntervals"]:
            if interval["provenance"].get("reviewedRangeBasis") == "legacyReviewRegistry":
                interval["provenance"]["reviewedRangeBasis"] = "catalogReviewedIntervals"
    for source in result["publicSources"]:
        destination = f"public/{source['sourceId']}"
        source["recordNamespace"] = source.get("recordNamespace", source["imageRoot"])
        source["metadataFiles"] = [f"{destination}/{Path(path).relative_to(source['imageRoot']).as_posix()}"
                                   for path in source["metadataFiles"]]
        source.update(imageRoot=destination, labelRoot=destination)
        source.pop("labelMapEvidence", None)
    return result


def rawImportEntries(catalog, sourceRoot):
    entries = {}

    def include(original, target, sourceId, role):
        target = relativePathValue(target)
        if target in entries and entries[target]["original"] != original:
            raise ValueError(f"Raw source destination collision: {target}.")
        originalPath = sourceRoot / relativePathValue(original)
        path = resolveSourcePath(sourceRoot, original)
        if originalPath.is_symlink() or not path.is_file():
            raise ValueError(f"Raw source must be an ordinary file: {original}.")
        entries[target] = {"original": original, "path": target, "sourceId": sourceId,
                           "role": role, "byteSize": path.stat().st_size}

    for video in catalog.data["ownVideos"]:
        for field, folder in (("videoPath", "videos"), ("annotationPath", "annotations")):
            include(video[field], f"own/{folder}/{Path(video[field]).name}", video["videoId"], folder)
    for source in catalog.data["publicSources"]:
        destination = f"public/{source['sourceId']}"
        for field, extensions, role in (("imageRoot", set(source["imageExtensions"]), "image"),
                                        ("labelRoot", {".txt"}, "annotation")):
            sourcePath = resolveSourcePath(sourceRoot, source[field])
            for path in scanFiles(sourcePath):
                if path.suffix.lower() in extensions:
                    relative = path.relative_to(sourceRoot).as_posix()
                    include(relative, f"{destination}/{path.relative_to(sourcePath).as_posix()}", source["sourceId"], role)
        for relative in source["metadataFiles"]:
            target = f"{destination}/{Path(relative).relative_to(source['imageRoot']).as_posix()}"
            include(relative, target, source["sourceId"], "sourceMetadata")
    return sorted(entries.values(), key=lambda item: item["path"])


def copyRawFile(source, target):
    import hashlib

    before = source.stat()
    digest = hashlib.sha256()
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as original, target.open("xb") as destination:
        while chunk := original.read(1024 * 1024):
            digest.update(chunk)
            destination.write(chunk)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Raw input changed during import: {source.name}.")
    return digest.hexdigest()


def importRawSources(catalog, sourceRoot, outputDir, progress=print):
    sourceRoot = Path(sourceRoot).resolve()
    outputDir = Path(outputDir).resolve()
    project = Path(__file__).resolve().parents[3]
    protected = [sourceRoot, project / "src", project / "configs", project / "data" / "prepared",
                 project / "data" / "projections", project / "data" / "modelWeights"]
    for path in protected:
        validateNewOutput(outputDir, path, protected)
    data = nativeCatalog(catalog.data)
    validateSources(data, catalog.labels)
    entries = rawImportEntries(catalog, sourceRoot)
    requireDiskSpace(outputDir, sum(item["byteSize"] for item in entries))
    with stagedDirectory(outputDir, sourceRoot) as staging:
        artifacts = []
        for index, entry in enumerate(entries, start=1):
            digest = copyRawFile(resolveSourcePath(sourceRoot, entry["original"]), staging / entry["path"])
            artifacts.append({key: value for key, value in entry.items() if key != "original"} | {"sha256": digest})
            if index % 1000 == 0 or index == len(entries):
                progress(f"Imported raw files: {index}/{len(entries)}")
        aliases = {item["original"]: item["path"] for item in entries}
        manifest = {"schemaVersion": 1, "kind": "rawDatasetSources", "sourcePolicy": "realOnly",
                    "catalog": data, "sourceAliases": aliases, "artifacts": artifacts}
        manifest["contractId"] = hashJson(manifest)
        writeJson(staging / "sourceManifest.json", manifest)
        verifyRawSources(staging)
        from futsal.datasets.datasetFiles import verifyEvidence

        verifyEvidence(sourceRoot, [{"path": entry["original"], "byteSize": artifact["byteSize"],
                                    "sha256": artifact["sha256"]}
                                   for entry, artifact in zip(entries, artifacts, strict=True)])
    return manifest


def verifyRawSources(root):
    from futsal.datasets.datasetFiles import verifyEvidence

    root = Path(root).resolve()
    manifestPath = root / "sourceManifest.json"
    metadata = manifestPath.stat()
    sourceAliases(manifestPath, metadata.st_mtime_ns, metadata.st_size)
    manifest = readJson(manifestPath)
    if manifest["contractId"] != hashJson({key: value for key, value in manifest.items() if key != "contractId"}):
        raise ValueError("Raw source manifest identity changed.")
    expected = {item["path"] for item in manifest["artifacts"]}
    files = scanFiles(root)
    if any(path.is_symlink() for path in files):
        raise ValueError("Raw source files must not be links.")
    observed = {path.relative_to(root).as_posix() for path in files}
    if len(expected) != len(manifest["artifacts"]) or observed != expected | {"sourceManifest.json"}:
        raise ValueError("Raw source membership differs from its manifest.")
    if manifest["sourcePolicy"] != "realOnly" or manifest["catalog"]["syntheticSources"]:
        raise ValueError("Raw migration requires only registered real inputs.")
    if not set(manifest["sourceAliases"].values()) <= expected:
        raise ValueError("Raw source binding points outside the imported file set.")
    verifyEvidence(root, manifest["artifacts"])
    return manifest
