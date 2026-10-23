"""Validate source declarations without opening or modifying source datasets."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import yaml


@dataclass(slots=True)
class Catalog:
    """Validated declarations and hashes of the exact configuration bytes read."""

    data: dict
    labels: dict
    configHashes: list[dict]


class StrictLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate and non-string mapping keys."""


def constructMapping(loader: StrictLoader, node: yaml.MappingNode) -> dict:
    loader.flatten_mapping(node)
    result = {}
    for keyNode, valueNode in node.value:
        key = loader.construct_object(keyNode, deep=True)
        if not isinstance(key, str):
            raise ValueError("Configuration mapping keys must be strings.")
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key!r}.")
        result[key] = loader.construct_object(valueNode, deep=True)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, constructMapping)


def hashFile(path: Path) -> str:
    """Compute SHA-256 in bounded memory without changing the input file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as sourceFile:
        for chunk in iter(lambda: sourceFile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def textValue(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected nonempty text.")
    return value


def relativePathValue(value: object, context: str) -> str:
    text = textValue(value, context)
    windowsPath = PureWindowsPath(text)
    normalized = text.replace("\\", "/")
    if (
        windowsPath.drive
        or windowsPath.root
        or PurePosixPath(normalized).is_absolute()
        or "\x00" in text
        or ":" in text
    ):
        raise ValueError(f"{context}: expected a relative path within the declared root.")
    depth = 0
    for part in PurePosixPath(normalized).parts:
        if part == "..":
            depth -= 1
            if depth < 0:
                raise ValueError(f"{context}: path escapes its declared root.")
        elif part != ".":
            depth += 1
    return normalized


def resolveSourcePath(dataRoot: Path, relativePath: str) -> Path:
    """Resolve a root-contained path, including symlinks and missing final files."""

    relativePath = relativePathValue(relativePath, "source path")
    try:
        rootPath = Path(dataRoot).resolve()
        sourcePath = (rootPath / relativePath).resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError("Unable to resolve source path safely.") from error
    if not sourcePath.is_relative_to(rootPath):
        raise ValueError("Source path escapes its declared root through a symbolic link.")
    return sourcePath


def mappingValue(value: object, context: str, required: set, optional: set | None = None) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected a mapping.")
    missing = required - value.keys()
    unknown = value.keys() - required - (optional or set())
    if missing or unknown:
        raise ValueError(f"{context}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}.")
    return value


def listValue(value: object, context: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected a list.")
    return value


def integerValue(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{context}: expected a nonnegative integer, not a boolean.")
    return value


def requireChoice(value: object, choices: set, context: str) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{context}: expected one of {sorted(choices)}.")


def uniqueText(value: object, seen: set, context: str) -> str:
    text = textValue(value, context)
    normalized = text.strip().casefold()
    if normalized in seen:
        raise ValueError(f"{context}: duplicate identifier or normalized label {text!r}.")
    seen.add(normalized)
    return text


def stringListValue(value: object, context: str, choices: set | None = None) -> list[str]:
    seen = set()
    for entry in listValue(value, context):
        uniqueText(entry, seen, context)
        if choices is not None:
            requireChoice(entry, choices, context)
    return value


def readYaml(path: Path) -> tuple[dict, dict]:
    try:
        content = path.read_bytes()
        document = yaml.load(content.decode("utf-8"), Loader=StrictLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"{path.name}: unable to read a valid UTF-8 YAML configuration.") from error
    if not isinstance(document, dict):
        raise ValueError(f"{path.name}: expected a YAML mapping.")
    if type(document.get("schemaVersion")) is not int or document["schemaVersion"] != 1:
        raise ValueError(f"{path.name}: schemaVersion must be integer 1.")
    fingerprint = {
        "path": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "byteSize": len(content),
    }
    return document, fingerprint


def validateLabels(labels: dict) -> set[str]:
    mappingValue(labels, "labels", {"schemaVersion", "canonicalClasses", "labelMaps"})
    classIds = set()
    normalizedNames = set()
    classNames = set()
    for entry in listValue(labels["canonicalClasses"], "canonicalClasses"):
        mappingValue(entry, "canonical class", {"classId", "name", "geometry"})
        classId = integerValue(entry["classId"], "classId")
        if classId in classIds:
            raise ValueError(f"canonicalClasses: duplicate classId {classId}.")
        classIds.add(classId)
        classNames.add(uniqueText(entry["name"], normalizedNames, "canonical class name"))
        requireChoice(entry["geometry"], {"boundingBox", "point"}, "canonical geometry")
    if classIds != set(range(25)):
        raise ValueError("canonicalClasses must contain exactly the class IDs 0 through 24.")
    labelMaps = labels["labelMaps"]
    if not isinstance(labelMaps, dict) or not labelMaps:
        raise ValueError("labelMaps must be a nonempty mapping.")
    mapIds = set()
    for mapId, labelMap in labelMaps.items():
        uniqueText(mapId, mapIds, "labelMapId")
        mappingValue(labelMap, mapId, {"inputType", "entries"}, {"normalization"})
        requireChoice(labelMap["inputType"], {"labelName", "classId"}, f"{mapId}.inputType")
        if labelMap["inputType"] == "labelName" or "normalization" in labelMap:
            requireChoice(labelMap.get("normalization"), {"stripCasefold"}, f"{mapId}.normalization")
        sourceLabels = set()
        sourceIds = set()
        entries = listValue(labelMap["entries"], f"{mapId}.entries")
        if not entries:
            raise ValueError(f"{mapId}: entries must not be empty.")
        for entry in entries:
            required = {"sourceLabel", "targetClass", "action"}
            if labelMap["inputType"] == "classId":
                required.add("sourceClassId")
            mappingValue(entry, f"{mapId} entry", required, {"semanticClass", "reason"})
            uniqueText(entry["sourceLabel"], sourceLabels, f"{mapId}.sourceLabel")
            if labelMap["inputType"] == "classId":
                sourceId = integerValue(entry["sourceClassId"], f"{mapId}.sourceClassId")
                if sourceId in sourceIds:
                    raise ValueError(f"{mapId}: duplicate sourceClassId {sourceId}.")
                sourceIds.add(sourceId)
            requireChoice(entry["action"], {"map", "ignore", "preserve"}, f"{mapId}.action")
            if entry["action"] == "map":
                requireChoice(entry["targetClass"], classNames, f"{mapId}.targetClass")
            elif entry["targetClass"] is not None:
                raise ValueError(f"{mapId}: ignore/preserve entries must have null targetClass.")
            else:
                textValue(entry.get("semanticClass"), f"{mapId}.semanticClass")
                textValue(entry.get("reason"), f"{mapId}.reason")
            for fieldName in ("semanticClass", "reason"):
                if fieldName in entry:
                    textValue(entry[fieldName], f"{mapId}.{fieldName}")
    return classNames


def validateMapReference(source: dict, labels: dict, inputType: str, context: str) -> None:
    mapId = textValue(source["labelMapId"], f"{context}.labelMapId")
    labelMap = labels["labelMaps"].get(mapId)
    if labelMap is None or labelMap["inputType"] != inputType:
        raise ValueError(f"{context}: labelMapId must reference a {inputType} label map.")


def validateOwnVideos(data: dict, labels: dict, sourceIds: set) -> None:
    intervalIds = set()
    matchIdsByVideo = {}
    matchReferences = []
    classNames = {entry["name"] for entry in labels["canonicalClasses"]}
    for video in listValue(data["ownVideos"], "ownVideos"):
        mappingValue(video, "own video", {
            "videoId", "videoPath", "annotationPath", "labelMapId", "legacyMatchId",
            "reviewedIntervals", "annotationScope", "classesAnnotated", "annotationScopeEvidence",
        })
        videoId = uniqueText(video["videoId"], sourceIds, "videoId")
        matchIdsByVideo[videoId] = set()
        requireChoice(video["annotationScope"], {
            "completeForDeclaredClasses", "partialForDeclaredClasses", "unknown",
        }, f"{videoId}.annotationScope")
        stringListValue(video["classesAnnotated"], f"{videoId}.classesAnnotated", classNames)
        textValue(video["annotationScopeEvidence"], f"{videoId}.annotationScopeEvidence")
        for fieldName in ("videoPath", "annotationPath"):
            relativePathValue(video[fieldName], f"{videoId}.{fieldName}")
        if video["legacyMatchId"] is not None:
            textValue(video["legacyMatchId"], f"{videoId}.legacyMatchId")
        validateMapReference(video, labels, "labelName", videoId)
        ranges = []
        intervals = listValue(video["reviewedIntervals"], f"{videoId}.reviewedIntervals")
        if not intervals:
            raise ValueError(f"{videoId}: reviewedIntervals must not be empty.")
        for interval in intervals:
            mappingValue(interval, "reviewed interval", {
                "intervalId", "matchId", "startFrame", "endFrame", "internalCutsStatus",
                "provenance",
            })
            uniqueText(interval["intervalId"], intervalIds, "intervalId")
            matchId = textValue(interval["matchId"], "matchId")
            matchIdsByVideo[videoId].add(matchId)
            requireChoice(interval["internalCutsStatus"], {"pending"}, "internalCutsStatus")
            provenance = mappingValue(interval["provenance"], "interval provenance", {
                "matchIdentityBasis", "reviewedRangeBasis",
            }, {"sameMatchAsVideoId"})
            for fieldName in ("matchIdentityBasis", "reviewedRangeBasis"):
                textValue(provenance[fieldName], f"provenance.{fieldName}")
            if "sameMatchAsVideoId" in provenance:
                referenceId = textValue(provenance["sameMatchAsVideoId"], "sameMatchAsVideoId")
                matchReferences.append((referenceId, matchId))
            startFrame = integerValue(interval["startFrame"], "startFrame")
            endFrame = integerValue(interval["endFrame"], "endFrame")
            if endFrame < startFrame:
                raise ValueError(f"{videoId}: endFrame precedes startFrame.")
            ranges.append((startFrame, endFrame))
        ranges.sort()
        if any(
            current[0] <= previous[1]
            for previous, current in zip(ranges, ranges[1:], strict=False)
        ):
            raise ValueError(f"{videoId}: reviewed intervals overlap (inclusive endpoints).")
    for referenceId, matchId in matchReferences:
        if matchId not in matchIdsByVideo.get(referenceId, set()):
            raise ValueError("sameMatchAsVideoId must reference a video containing the same matchId.")


def validateYoloSources(data: dict, labels: dict, sourceIds: set) -> None:
    for category in ("publicSources", "syntheticSources"):
        for source in listValue(data[category], category):
            mappingValue(source, category, {
                "sourceId", "sourceName", "format", "required", "imageRoot", "labelRoot",
                "labelMapId", "annotationScope", "allowedUses", "imageExtensions",
                "annotationIgnorePatterns", "metadataFiles", "labelMapEvidence",
            })
            sourceId = uniqueText(source["sourceId"], sourceIds, "sourceId")
            textValue(source["sourceName"], f"{sourceId}.sourceName")
            requireChoice(source["format"], {"yolo", "yoloPoints"}, f"{sourceId}.format")
            if type(source["required"]) is not bool:
                raise ValueError(f"{sourceId}.required must be a boolean.")
            for fieldName in ("imageRoot", "labelRoot", "labelMapEvidence"):
                relativePathValue(source[fieldName], f"{sourceId}.{fieldName}")
            textValue(source["annotationScope"], f"{sourceId}.annotationScope")
            uses = stringListValue(source["allowedUses"], f"{sourceId}.allowedUses", {"training"})
            if not uses:
                raise ValueError(f"{sourceId}: allowedUses must include training.")
            extensions = stringListValue(source["imageExtensions"], f"{sourceId}.imageExtensions")
            if not extensions or any(
                not extension.startswith(".") or not extension[1:].isalnum()
                for extension in extensions
            ):
                raise ValueError(f"{sourceId}: imageExtensions must contain dotted extensions.")
            stringListValue(source["annotationIgnorePatterns"], f"{sourceId}.annotationIgnorePatterns")
            metadataFiles = stringListValue(source["metadataFiles"], f"{sourceId}.metadataFiles")
            for metadataPath in metadataFiles:
                relativePathValue(metadataPath, f"{sourceId}.metadataFiles")
            validateMapReference(source, labels, "classId", sourceId)


def validatePolicy(policy: object, classNames: set) -> None:
    mappingValue(policy, "splitPolicy", {
        "selectionStatus", "groupBy", "countingUnit", "targetFractions", "rareClassRequirements",
    })
    requireChoice(policy["selectionStatus"], {"unassigned"}, "splitPolicy.selectionStatus")
    requireChoice(policy["groupBy"], {"matchId"}, "splitPolicy.groupBy")
    requireChoice(policy["countingUnit"], {"reviewedFrameRecord"}, "splitPolicy.countingUnit")
    splitNames = {"training", "validation", "test"}
    fractions = mappingValue(policy["targetFractions"], "targetFractions", splitNames)
    for value in fractions.values():
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("targetFractions must contain finite numbers between zero and one.")
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("targetFractions must sum to one.")
    seen = set()
    for requirement in listValue(policy["rareClassRequirements"], "rareClassRequirements"):
        mappingValue(requirement, "rare class requirement", {
            "className", "requiredSplits", "unsupportedSplits", "reason",
        })
        uniqueText(requirement["className"], seen, "rare className")
        requireChoice(requirement["className"], classNames, "rare className")
        required = stringListValue(requirement["requiredSplits"], "requiredSplits", splitNames)
        unsupported = stringListValue(requirement["unsupportedSplits"], "unsupportedSplits", splitNames)
        if set(required) & set(unsupported):
            raise ValueError("A split cannot be both required and unsupported for the same class.")
        textValue(requirement["reason"], "rare class reason")


def loadCatalog(configPath: Path) -> Catalog:
    """Load schema version 1; source existence and observed counts belong to inventory."""

    configPath = Path(configPath)
    data, sourceHash = readYaml(configPath)
    mappingValue(data, "catalog", {
        "schemaVersion", "labelMapPath", "legacyReviewPath", "sourceAccess", "expected",
        "splitPolicy", "ownVideos", "publicSources", "syntheticSources",
    })
    labelPath = resolveSourcePath(configPath.parent, data["labelMapPath"])
    labels, labelHash = readYaml(labelPath)
    classNames = validateLabels(labels)
    relativePathValue(data["legacyReviewPath"], "legacyReviewPath")
    requireChoice(data["sourceAccess"], {"readOnly"}, "sourceAccess")
    expected = mappingValue(data["expected"], "expected", {
        "reviewedFrames", "matchCount", "videoCount", "reviewedIntervalCount",
    })
    for fieldName, value in expected.items():
        integerValue(value, f"expected.{fieldName}")
    validatePolicy(data["splitPolicy"], classNames)
    sourceIds = set()
    validateOwnVideos(data, labels, sourceIds)
    validateYoloSources(data, labels, sourceIds)
    return Catalog(data=data, labels=labels, configHashes=[sourceHash, labelHash])
