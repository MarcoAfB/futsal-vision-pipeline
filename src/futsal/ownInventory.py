from collections import Counter, defaultdict
from pathlib import Path

import yaml

from futsal.annotationAudit import addIssue, auditShape
from futsal.catalog import Catalog, StrictLoader, hashFile, resolveSourcePath
from futsal.cvatReader import readCvat
from futsal.mediaMetadata import readVideoMetadata


def fileEvidence(dataRoot: Path, relativePath: str) -> dict:
    path = resolveSourcePath(dataRoot, relativePath)
    return {"path": relativePath, "sha256": hashFile(path), "byteSize": path.stat().st_size}


def fileError(error: OSError | ValueError) -> str:
    if isinstance(error, OSError):
        return error.strerror or str(error)
    return str(error)


def readReviewRegistry(catalog: Catalog, dataRoot: Path, issues: list[dict]) -> tuple[dict, dict]:
    relativePath = catalog.data["legacyReviewPath"]
    try:
        evidence = fileEvidence(dataRoot, relativePath)
        registryPath = resolveSourcePath(dataRoot, relativePath)
        registry = yaml.load(registryPath.read_text(encoding="utf-8-sig"), Loader=StrictLoader)
        if not isinstance(registry, dict) or not isinstance(registry.get("partidas"), dict):
            raise ValueError("The legacy review registry must contain a partidas mapping.")
        return registry["partidas"], evidence
    except (OSError, ValueError, yaml.YAMLError) as error:
        addIssue(
            issues,
            "error",
            "reviewRegistryUnreadable",
            f"Cannot verify authoritative reviewed limits: {fileError(error)}",
            sourcePath=relativePath,
        )
        return {}, {"path": relativePath, "status": "unavailable"}


def verifyReviewedLimits(video: dict, registry: dict, issues: list[dict]) -> bool:
    legacyMatchId = video.get("legacyMatchId")
    if legacyMatchId is None:
        return True
    entry = registry.get(legacyMatchId)
    intervals = video["reviewedIntervals"]
    valid = (
        isinstance(entry, dict)
        and type(entry.get("maximoFrames")) is int
        and entry["maximoFrames"] > 0
        and entry.get("video") == video["videoPath"]
        and entry.get("xml") == video["annotationPath"]
        and len(intervals) == 1
        and intervals[0]["startFrame"] == 0
        and intervals[0]["endFrame"] == entry["maximoFrames"] - 1
    )
    if not valid:
        addIssue(
            issues,
            "error",
            "reviewRegistryMismatch",
            "Catalog paths or reviewed interval disagree with the authoritative registry; "
            "this video's frames are not included in the eligible inventory.",
            sourceId=video["videoId"],
            legacyMatchId=legacyMatchId,
        )
    return valid


def inventoryOwnVideo(
    video: dict, catalog: Catalog, dataRoot: Path, registry: dict, issues: list[dict]
) -> dict:
    sourceId = video["videoId"]
    report = {
        "videoId": sourceId,
        "videoPath": video["videoPath"],
        "annotationPath": video["annotationPath"],
        "annotationScope": video["annotationScope"],
        "annotationScopeEvidence": video["annotationScopeEvidence"],
        "classesAnnotated": video["classesAnnotated"],
        "reviewedIntervals": video["reviewedIntervals"],
        "labelMapId": video["labelMapId"],
        "availability": "available",
        "frames": [],
    }
    if not verifyReviewedLimits(video, registry, issues):
        report["availability"] = "reviewLimitsUnverified"
        return report
    try:
        report["annotationEvidence"] = fileEvidence(dataRoot, video["annotationPath"])
        document = readCvat(resolveSourcePath(dataRoot, video["annotationPath"]))
    except (OSError, ValueError) as error:
        addIssue(issues, "error", "annotationUnreadable", fileError(error), sourceId=sourceId)
        report["availability"] = "annotationsUnavailable"
        return report
    report["annotationMetadata"] = document.metadata
    report["declaredLabels"] = document.declaredLabels
    report["outsideShapeCountInFile"] = document.outsideShapeCount
    if document.metadata.get("annotationMode") in {"images", "mixed"}:
        addIssue(
            issues,
            "error",
            "videoFrameMappingRequired",
            "CVAT image IDs cannot be assumed to be original video frame indices.",
            sourceId=sourceId,
        )
        report["availability"] = "frameMappingRequired"
        return report
    media, dimensions = readOwnMedia(video, document, dataRoot, issues)
    report["media"] = media
    if media.get("status") == "unavailable":
        report["availability"] = "mediaUnavailable"
    frames = reviewedFrames(video, document, media, dimensions, catalog, issues)
    reviewedShapes = sum(len(frame["annotations"]) for frame in frames)
    report.update(
        {
            "frames": frames,
            "reviewedFrameRecordCount": len(frames),
            "activeShapeCountInReviewedFrames": reviewedShapes,
            "activeShapeCountOutsideReviewedFrames": len(document.shapes) - reviewedShapes,
        }
    )
    return report


def readOwnMedia(video, document, dataRoot, issues):
    sourceId = video["videoId"]
    try:
        media = readVideoMetadata(resolveSourcePath(dataRoot, video["videoPath"]))
    except (OSError, ValueError) as error:
        addIssue(issues, "error", "videoUnreadable", fileError(error), sourceId=sourceId)
        media = {"status": "unavailable"}
    fallbackDimensions = document.defaultDimensions
    if media.get("width") and media.get("height"):
        mediaDimensions = (media["width"], media["height"])
        if fallbackDimensions is not None and fallbackDimensions != mediaDimensions:
            addIssue(
                issues,
                "error",
                "dimensionMismatch",
                "CVAT and video dimensions differ; annotations retain CVAT coordinates.",
                sourceId=sourceId,
            )
        if fallbackDimensions is None:
            fallbackDimensions = mediaDimensions
    return media, fallbackDimensions


def reviewedFrames(video, document, media, fallbackDimensions, catalog, issues):
    labelEntries = {
        entry["sourceLabel"].strip().casefold(): entry
        for entry in catalog.labels["labelMaps"][video["labelMapId"]]["entries"]
    }
    classGeometry = {
        entry["name"]: entry["geometry"] for entry in catalog.labels["canonicalClasses"]
    }
    shapesByFrame = defaultdict(list)
    for shape in document.shapes:
        shapesByFrame[shape.frameIndex].append(shape)
    frames = []
    for interval in video["reviewedIntervals"]:
        if media.get("frameCount") and interval["endFrame"] >= media["frameCount"]:
            addIssue(
                issues,
                "error",
                "reviewIntervalBeyondMedia",
                "Reviewed interval exceeds the reported video frame count; no range was changed.",
                sourceId=video["videoId"],
                intervalId=interval["intervalId"],
            )
        for frameIndex in range(interval["startFrame"], interval["endFrame"] + 1):
            context = {
                "sourceId": video["videoId"],
                "frameIndex": frameIndex,
                "matchId": interval["matchId"],
                "intervalId": interval["intervalId"],
            }
            dimensions = document.dimensions.get(frameIndex, fallbackDimensions)
            frames.append(
                auditFrame(
                    context,
                    shapesByFrame[frameIndex],
                    labelEntries,
                    classGeometry,
                    dimensions,
                    media.get("fps"),
                    issues,
                )
            )
    return frames


def auditFrame(context, shapes, labelEntries, classGeometry, dimensions, fps, issues):
    annotations = [auditShape(shape, labelEntries, dimensions, context, issues) for shape in shapes]
    classCounts = Counter(
        item["canonicalClass"] for item in annotations if item["mappingAction"] == "map"
    )
    for className, count in classCounts.items():
        if count > 1 and (className == "ball" or classGeometry[className] == "point"):
            code = "multipleBallAnnotations" if className == "ball" else "duplicateLandmark"
            addIssue(
                issues,
                "warning",
                code,
                "Multiple annotations require review; all observations were preserved.",
                canonicalClass=className,
                annotationCount=count,
                **context,
            )
    return {
        "recordId": f"{context['sourceId']}:{context['frameIndex']}",
        "frameIndex": context["frameIndex"],
        "intervalId": context["intervalId"],
        "matchId": context["matchId"],
        "videoTimeSeconds": context["frameIndex"] / fps if fps else None,
        "originalMatchTimeSeconds": None,
        "dimensions": list(dimensions) if dimensions else None,
        "coordinateSystem": "imagePixels",
        "classCounts": dict(sorted(classCounts.items())),
        "annotations": annotations,
    }
