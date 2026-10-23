import fnmatch
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from futsal.catalog import hashFile, resolveSourcePath
from futsal.mediaMetadata import readImageMetadata
from futsal.yoloAnnotations import readLabels


def issue(issues: list, source: dict, severity: str, code: str, message: str, **details):
    issues.append(
        {
            "severity": severity,
            "code": code,
            "sourceId": source["sourceId"],
            "message": message,
            **details,
        }
    )


def scanFiles(rootPath: Path) -> list[Path]:

    found = []
    pending = [rootPath]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink() and entry.is_dir():
                    raise ValueError("Directory symbolic links are not supported in source scans.")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file() or entry.is_symlink():
                    found.append(path)
    return sorted(found, key=lambda path: (path.as_posix().casefold(), path.as_posix()))


def sourceRelativePath(path: Path, dataRoot: Path) -> str:
    return path.relative_to(dataRoot).as_posix()


def fingerprint(dataRoot: Path, relativePath: str) -> dict:
    path = resolveSourcePath(dataRoot, relativePath)
    return {"path": relativePath, "sha256": hashFile(path), "byteSize": path.stat().st_size}


def readEvidence(source: dict, dataRoot: Path, report: dict, issues: list) -> None:
    severity = "error" if source["required"] else "warning"
    for relativePath in source["metadataFiles"]:
        try:
            report["metadataFiles"].append(fingerprint(dataRoot, relativePath))
        except (OSError, ValueError) as error:
            issue(
                issues,
                source,
                severity,
                "metadataUnavailable",
                f"Cannot read registered metadata ({type(error).__name__}).",
                path=relativePath,
            )
    relativePath = source["labelMapEvidence"]
    try:
        report["labelMapEvidence"] = fingerprint(dataRoot, relativePath)
    except (OSError, ValueError) as error:
        issue(
            issues,
            source,
            severity,
            "labelMapEvidenceUnavailable",
            f"Cannot read label mapping evidence ({type(error).__name__}).",
            path=relativePath,
        )


def checkDocumentedCount(source: dict, dataRoot: Path, report: dict, issues: list) -> None:
    for metadata in report["metadataFiles"]:
        if Path(metadata["path"]).suffix.casefold() != ".txt":
            continue
        try:
            text = resolveSourcePath(dataRoot, metadata["path"]).read_text(encoding="utf-8-sig")
        except (OSError, ValueError, UnicodeError) as error:
            issue(
                issues,
                source,
                "warning",
                "metadataTextUnreadable",
                f"Cannot inspect metadata text ({type(error).__name__}).",
                path=metadata["path"],
            )
            continue
        documentedCounts = set()
        for value in re.findall(r"\b(\d[\d,]*)\s+images\b", text, flags=re.IGNORECASE):
            try:
                documentedCounts.add(int(value.replace(",", "")))
            except ValueError:
                issue(
                    issues,
                    source,
                    "warning",
                    "invalidDocumentedImageCount",
                    "The metadata image count exceeds supported integer size.",
                    path=metadata["path"],
                )
        if len(documentedCounts) == 1:
            documentedCount = documentedCounts.pop()
            if documentedCount != report["imageCount"]:
                issue(
                    issues,
                    source,
                    "warning",
                    "documentedImageCountMismatch",
                    "The image count stated by source metadata differs from observed files.",
                    path=metadata["path"],
                    documentedImageCount=documentedCount,
                    observedImageCount=report["imageCount"],
                )


def stemGroups(paths: list[Path], rootPath: Path) -> dict[str, list[Path]]:
    groups = defaultdict(list)
    for path in paths:
        groups[path.relative_to(rootPath).with_suffix("").as_posix().casefold()].append(path)
    return dict(groups)


def inventoryYoloSource(source: dict, labelMap: dict, dataRoot: Path) -> tuple[dict, list[dict]]:

    dataRoot = Path(dataRoot).resolve()
    issues = []
    report = {
        "sourceId": source["sourceId"],
        "sourceName": source["sourceName"],
        "format": source["format"],
        "annotationScope": source["annotationScope"],
        "allowedUses": list(source["allowedUses"]),
        "availability": "available",
        "imageCount": 0,
        "annotationFileCount": 0,
        "emptyLabelCount": 0,
        "sourceClassCounts": {},
        "canonicalClassCounts": {},
        "samples": [],
        "annotationFiles": [],
        "metadataFiles": [],
        "labelMapEvidence": None,
    }
    readEvidence(source, dataRoot, report, issues)
    try:
        imageRoot = resolveSourcePath(dataRoot, source["imageRoot"])
        labelRoot = resolveSourcePath(dataRoot, source["labelRoot"])
        imageFiles = scanFiles(imageRoot)
        labelFiles = imageFiles if labelRoot == imageRoot else scanFiles(labelRoot)
    except (OSError, ValueError) as error:
        report["availability"] = "unavailable"
        issue(
            issues,
            source,
            "error" if source["required"] else "warning",
            "sourceUnavailable",
            f"Cannot completely enumerate registered source roots ({type(error).__name__}).",
        )
        return report, issues
    extensions = {extension.casefold() for extension in source["imageExtensions"]}
    imagePaths = [path for path in imageFiles if path.suffix.casefold() in extensions]
    ignorePatterns = [pattern.casefold() for pattern in source["annotationIgnorePatterns"]]
    labelPaths = [
        path
        for path in labelFiles
        if path.suffix.casefold() == ".txt"
        and not any(
            fnmatch.fnmatchcase(path.name.casefold(), pattern)
            or fnmatch.fnmatchcase(path.relative_to(labelRoot).as_posix().casefold(), pattern)
            for pattern in ignorePatterns
        )
    ]
    report["imageCount"] = len(imagePaths)
    report["annotationFileCount"] = len(labelPaths)
    if not imagePaths:
        report["availability"] = "empty"
        issue(
            issues,
            source,
            "error" if source["required"] else "warning",
            "noSourceImages",
            "No registered image files were found; this is not a populated source.",
        )
    imageGroups = stemGroups(imagePaths, imageRoot)
    labelGroups = stemGroups(labelPaths, labelRoot)
    for groups, collisionCode in (
        (imageGroups, "ambiguousImageStem"),
        (labelGroups, "ambiguousLabelStem"),
    ):
        for paths in groups.values():
            if len(paths) > 1:
                issue(
                    issues,
                    source,
                    "error",
                    collisionCode,
                    "Files share a relative stem; no unique image-label pairing can be selected.",
                    paths=[sourceRelativePath(path, dataRoot) for path in paths],
                )
    labelEntries = {entry["sourceClassId"]: entry for entry in labelMap["entries"]}
    labelsByPath = inventoryLabelFiles(
        labelPaths, labelRoot, imageGroups, labelEntries, source, dataRoot, report, issues
    )
    inventoryImages(
        imagePaths,
        imageRoot,
        imageGroups,
        labelGroups,
        labelsByPath,
        source,
        dataRoot,
        report,
        issues,
    )
    checkDocumentedCount(source, dataRoot, report, issues)
    if report["availability"] == "available" and any(
        issue["severity"] == "error" for issue in issues
    ):
        report["availability"] = "partial"
    return report, issues


def inventoryLabelFiles(
    labelPaths, labelRoot, imageGroups, labelEntries, source, dataRoot, report, issues
):
    labelsByPath = {}
    sourceCounts = Counter()
    canonicalCounts = Counter()
    for path in labelPaths:
        relativePath = sourceRelativePath(path, dataRoot)
        labels = readLabels(relativePath, source, labelEntries, dataRoot, issues)
        labelsByPath[path] = labels
        report["annotationFiles"].append(labels)
        report["emptyLabelCount"] += labels["empty"] is True
        sourceCounts.update(labels["sourceClassCounts"])
        canonicalCounts.update(labels["canonicalClassCounts"])
        stem = path.relative_to(labelRoot).with_suffix("").as_posix().casefold()
        if stem not in imageGroups:
            issue(
                issues,
                source,
                "error",
                "orphanLabel",
                "Annotation file has no corresponding registered image.",
                labelPath=relativePath,
            )
    if report["emptyLabelCount"]:
        issue(
            issues,
            source,
            "warning",
            "emptyLabels",
            "Empty annotation files do not establish true-negative images.",
            emptyLabelCount=report["emptyLabelCount"],
        )
    report["sourceClassCounts"] = dict(sorted(sourceCounts.items()))
    report["canonicalClassCounts"] = dict(sorted(canonicalCounts.items()))
    return labelsByPath


def inventoryImages(
    imagePaths, imageRoot, imageGroups, labelGroups, labelsByPath, source, dataRoot, report, issues
):
    for path in imagePaths:
        relativePath = sourceRelativePath(path, dataRoot)
        sample = {
            "recordId": f"{source['sourceId']}:{relativePath}",
            "imagePath": relativePath,
            "labelPath": None,
            "labelSha256": None,
            "media": None,
            "sourceClassCounts": {},
            "canonicalClassCounts": {},
        }
        try:
            sample["media"] = readImageMetadata(resolveSourcePath(dataRoot, relativePath))
        except (OSError, ValueError) as error:
            issue(
                issues,
                source,
                "error",
                "imageUnreadable",
                f"Cannot decode source image ({type(error).__name__}).",
                imagePath=relativePath,
            )
        stem = path.relative_to(imageRoot).with_suffix("").as_posix().casefold()
        matches = labelGroups.get(stem, [])
        if not matches:
            issue(
                issues,
                source,
                "error",
                "missingLabel",
                "Image has no corresponding annotation file; annotation scope is unknown.",
                imagePath=relativePath,
            )
        elif len(matches) == 1 and len(imageGroups[stem]) == 1:
            labels = labelsByPath[matches[0]]
            sample.update(
                {
                    "labelPath": labels["path"],
                    "labelSha256": labels["sha256"],
                    "sourceClassCounts": labels["sourceClassCounts"],
                    "canonicalClassCounts": labels["canonicalClassCounts"],
                }
            )
        report["samples"].append(sample)
