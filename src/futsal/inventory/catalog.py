import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path, PureWindowsPath

import yaml


@dataclass(slots=True)
class Catalog:
    data: dict
    labels: dict
    configHashes: list[dict]


class StrictLoader(yaml.SafeLoader):
    pass


def constructMapping(loader: StrictLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for keyNode, valueNode in node.value:
        key = loader.construct_object(keyNode)
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key!r}.")
        result[key] = loader.construct_object(valueNode)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, constructMapping)


def hashFile(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relativePathValue(value: str) -> str:
    path = PureWindowsPath(value)
    if path.drive or path.root or ".." in path.parts:
        raise ValueError("Source paths must be relative and stay within the declared root.")
    return value.replace("\\", "/")


def resolveSourcePath(dataRoot: Path, relativePath: str) -> Path:
    root = dataRoot.resolve()
    relativePath = relativePathValue(relativePath)
    path = (root / relativePath).resolve()
    manifestPath = root / "sourceManifest.json"
    if not path.exists() and manifestPath.is_file():
        metadata = manifestPath.stat()
        aliases = sourceAliases(manifestPath, metadata.st_mtime_ns, metadata.st_size)
        path = (root / relativePathValue(aliases.get(relativePath, relativePath))).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source path escapes its declared root through a symbolic link.")
    return path


@lru_cache(maxsize=8)
def sourceAliases(manifestPath, modifiedAt, byteSize):
    manifest = json.loads(manifestPath.read_text(encoding="utf-8"))
    payload = {key: value for key, value in manifest.items() if key != "contractId"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("kind") != "rawDatasetSources" or manifest.get("schemaVersion") != 1 or manifest.get("contractId") != digest:
        raise ValueError("Raw source manifest identity changed.")
    aliases = manifest["sourceAliases"]
    for original, target in aliases.items():
        relativePathValue(original)
        relativePathValue(target)
    return aliases


def readYaml(path: Path) -> tuple[dict, dict]:
    content = path.read_bytes()
    try:
        document = yaml.load(content.decode("utf-8-sig"), Loader=StrictLoader)
    except (UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"{path.name}: invalid UTF-8 YAML.") from error
    if not isinstance(document, dict):
        raise ValueError(f"{path.name}: expected a YAML mapping.")
    return document, {
        "path": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byteSize": len(content),
    }


def requireUnique(values, context: str) -> None:
    normalized = [str(value).strip().casefold() for value in values]
    if any(not value for value in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError(f"{context}: empty or duplicate identifier or normalized label.")


def validateLabelMap(labelMap: dict, classNames: set[str]) -> None:
    inputType = labelMap["inputType"]
    if inputType not in {"labelName", "classId"}:
        raise ValueError("Label maps must use labelName or classId inputs.")
    requireUnique([entry["sourceLabel"] for entry in labelMap["entries"]], "source labels")
    if inputType == "classId":
        ids = [entry["sourceClassId"] for entry in labelMap["entries"]]
        if any(type(value) is not int or value < 0 for value in ids):
            raise ValueError("Source class IDs must be nonnegative integers.")
        requireUnique(ids, "source class IDs")
    for entry in labelMap["entries"]:
        action, target = entry["action"], entry["targetClass"]
        if action == "map" and target not in classNames:
            raise ValueError(f"Unknown canonical target: {target!r}.")
        if action in {"ignore", "preserve"} and target is not None:
            raise ValueError("Ignored or generic labels cannot acquire a canonical target.")
        if action not in {"map", "ignore", "preserve"}:
            raise ValueError(f"Unknown mapping action: {action!r}.")


def validateLabels(labels: dict) -> None:
    classes = labels["canonicalClasses"]
    if [entry["classId"] for entry in classes] != list(range(25)):
        raise ValueError("Canonical class IDs must be ordered from 0 through 24.")
    requireUnique([entry["name"] for entry in classes], "canonical classes")
    if any(entry["geometry"] not in {"boundingBox", "point"} for entry in classes):
        raise ValueError("Canonical geometry must be boundingBox or point.")
    classNames = {entry["name"] for entry in classes}
    for labelMap in labels["labelMaps"].values():
        validateLabelMap(labelMap, classNames)


def validateReviewedIntervals(video: dict) -> None:
    ranges = []
    for interval in video["reviewedIntervals"]:
        start, end = interval["startFrame"], interval["endFrame"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ValueError("Reviewed ranges require nonnegative integer startFrame <= endFrame.")
        ranges.append((start, end))
    ranges.sort()
    if not ranges or any(current[0] <= previous[1] for previous, current in pairwise(ranges)):
        raise ValueError("Reviewed intervals must be present and must not overlap.")


def validateSourceMap(source: dict, labels: dict, inputType: str) -> None:
    labelMap = labels["labelMaps"].get(source["labelMapId"])
    if labelMap is None or labelMap["inputType"] != inputType:
        raise ValueError(f"Source requires a {inputType} label map.")


def validateSources(data: dict, labels: dict) -> None:
    videos = data["ownVideos"]
    external = data["publicSources"] + data["syntheticSources"]
    requireUnique(
        [video["videoId"] for video in videos] + [source["sourceId"] for source in external],
        "source IDs",
    )
    intervals = [interval for video in videos for interval in video["reviewedIntervals"]]
    requireUnique([interval["intervalId"] for interval in intervals], "interval IDs")
    matchesByVideo = {
        video["videoId"]: {item["matchId"] for item in video["reviewedIntervals"]}
        for video in videos
    }
    for video in videos:
        validateReviewedIntervals(video)
        validateSourceMap(video, labels, "labelName")
        for field in ("videoPath", "annotationPath"):
            relativePathValue(video[field])
        for interval in video["reviewedIntervals"]:
            reference = interval["provenance"].get("sameMatchAsVideoId")
            if reference is not None and interval["matchId"] not in matchesByVideo.get(
                reference, set()
            ):
                raise ValueError("sameMatchAsVideoId must identify the same original match.")
    for source in external:
        validateSourceMap(source, labels, "classId")
        if source["format"] != "yolo" or source["allowedUses"] != ["training"]:
            raise ValueError("External sources must use YOLO boxes and be training-only.")
        fields = ("imageRoot", "labelRoot", "labelMapEvidence") if data["schemaVersion"] == 1 else ("imageRoot", "labelRoot", "recordNamespace")
        for field in fields:
            relativePathValue(source[field])
        for metadataPath in source["metadataFiles"]:
            relativePathValue(metadataPath)


def loadCatalog(configPath: Path) -> Catalog:
    try:
        data, sourceHash = readYaml(configPath)
        labelPath = resolveSourcePath(configPath.parent, data["labelMapPath"])
        labels, labelHash = readYaml(labelPath)
        if data["schemaVersion"] not in {1, 2} or labels["schemaVersion"] != 1:
            raise ValueError("Unsupported source or label schema version.")
        validateLabels(labels)
        validateSources(data, labels)
        if data["schemaVersion"] == 1:
            relativePathValue(data["legacyReviewPath"])
        elif data.get("reviewPolicy") != "catalogReviewedIntervals" or "legacyReviewPath" in data or any("legacyMatchId" in video for video in data["ownVideos"]):
            raise ValueError("Native sources require the project's own reviewed interval declarations.")
        labelHash["path"] = relativePathValue(data["labelMapPath"])
        return Catalog(data, labels, [sourceHash, labelHash])
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"{configPath.name}: missing or invalid configuration field {error}."
        ) from error
