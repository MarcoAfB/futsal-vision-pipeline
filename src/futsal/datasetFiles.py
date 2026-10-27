import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from futsal.catalog import hashFile, resolveSourcePath


def hashJson(value):
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def writeJson(path, value):
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":"))
    Path(path).write_text(content + "\n", encoding="utf-8", newline="\n")


def readJson(path):
    def rejectConstant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}.")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=rejectConstant)


def validateNewOutput(outputDir, sourceRoot, protectedPaths=()):
    if Path(outputDir).is_symlink():
        raise ValueError("outputDir must not be a symbolic link.")
    outputDir, sourceRoot = Path(outputDir).resolve(), Path(sourceRoot).resolve()
    if outputDir.is_relative_to(sourceRoot):
        raise ValueError("outputDir must be outside dataRoot.")
    if outputDir.exists():
        raise ValueError("outputDir must be a new directory; overwriting is not supported.")
    if any(Path(path).resolve().is_relative_to(outputDir) for path in protectedPaths):
        raise ValueError("outputDir must not contain protected inputs.")
    return outputDir


@contextmanager
def stagedDirectory(outputDir, sourceRoot, protectedPaths=()):
    outputDir = validateNewOutput(outputDir, sourceRoot, protectedPaths)
    outputDir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{outputDir.name}.", dir=outputDir.parent))
    try:
        yield staging
        if outputDir.exists():
            raise ValueError("outputDir appeared during preparation; publication refused.")
        staging.rename(outputDir)
    finally:
        if staging.exists():
            if staging.resolve().parent != outputDir.parent.resolve():
                raise ValueError("Temporary directory escaped the intended output parent.")
            shutil.rmtree(staging)


def verifyEvidence(root, evidence):
    for item in evidence:
        path = resolveSourcePath(Path(root), item["path"])
        if path.stat().st_size != item["byteSize"] or hashFile(path) != item["sha256"]:
            raise ValueError(f"Input changed since planning: {item['path']}.")


def sourceEvidence(inventory, samples, dataRoot):
    expected = {}
    for video in inventory["ownVideos"]:
        expected[video["videoPath"]] = {"byteSize": video["media"]["byteSize"]}
        expected[video["annotationPath"]] = video["annotationEvidence"]
    registry = inventory["reviewRegistryEvidence"]
    expected[registry["path"]] = registry
    usedPublic = {sample["sourceId"] for sample in samples if sample["origin"] == "public"}
    for report in inventory["externalSources"]:
        if report["sourceId"] not in usedPublic:
            continue
        for item in report["annotationFiles"] + report["metadataFiles"]:
            if item.get("sha256"):
                expected[item["path"]] = item
        if report["labelMapEvidence"]:
            item = report["labelMapEvidence"]
            expected[item["path"]] = item
        for sample in report["samples"]:
            expected[sample["imagePath"]] = sample["media"] or {}
    evidence = []
    for relativePath, original in sorted(expected.items()):
        path = resolveSourcePath(Path(dataRoot), relativePath)
        observed = {"path": relativePath, "byteSize": path.stat().st_size, "sha256": hashFile(path)}
        if any(original.get(field) is not None and original[field] != observed[field]
               for field in ("byteSize", "sha256")):
            raise ValueError(f"Source changed during planning: {relativePath}.")
        evidence.append(observed)
    return evidence
