import platform
from collections import Counter
from pathlib import Path

from futsal import __version__
from futsal.annotationAudit import addIssue
from futsal.catalog import Catalog
from futsal.ownInventory import inventoryOwnVideo, readReviewRegistry
from futsal.yoloInventory import inventoryYoloSource


def summarizeOwnData(
    videos: list[dict], canonicalClasses: list[dict]
) -> tuple[list[dict], list[dict]]:
    matches: dict[str, dict] = {}
    for video in videos:
        for frame in video["frames"]:
            match = matches.setdefault(
                frame["matchId"],
                {
                    "matchId": frame["matchId"],
                    "reviewedFrameRecordCount": 0,
                    "videoIds": set(),
                    "intervalIds": set(),
                    "classAnnotationCounts": Counter(),
                    "classFrameCounts": Counter(),
                },
            )
            match["reviewedFrameRecordCount"] += 1
            match["videoIds"].add(video["videoId"])
            match["intervalIds"].add(frame["intervalId"])
            match["classAnnotationCounts"].update(frame["classCounts"])
            match["classFrameCounts"].update(frame["classCounts"].keys())
    matchReports = []
    classNames = [entry["name"] for entry in canonicalClasses]
    for matchId in sorted(matches):
        match = matches[matchId]
        matchReports.append(
            {
                **match,
                "videoIds": sorted(match["videoIds"]),
                "intervalIds": sorted(match["intervalIds"]),
                "classAnnotationCounts": {
                    name: match["classAnnotationCounts"][name] for name in classNames
                },
                "classFrameCounts": {name: match["classFrameCounts"][name] for name in classNames},
                "missingClasses": [
                    name for name in classNames if not match["classFrameCounts"][name]
                ],
            }
        )
    return matchReports, summarizeCoverage(matchReports, classNames)


def buildInventory(dataRoot: Path, catalog: Catalog) -> dict:

    dataRoot = Path(dataRoot).resolve()
    if not dataRoot.is_dir():
        raise ValueError("dataRoot must be an existing directory.")
    issues: list[dict] = []
    registry, registryEvidence = readReviewRegistry(catalog, dataRoot, issues)
    videos = [
        inventoryOwnVideo(video, catalog, dataRoot, registry, issues)
        for video in catalog.data["ownVideos"]
    ]
    matches, coverage = summarizeOwnData(videos, catalog.labels["canonicalClasses"])
    counts = {
        "reviewedFrames": sum(len(video["frames"]) for video in videos),
        "matchCount": len(matches),
        "videoCount": sum(bool(video["frames"]) for video in videos),
        "reviewedIntervalCount": len(
            {frame["intervalId"] for video in videos for frame in video["frames"]}
        ),
    }
    checkCoverage(counts, coverage, catalog.data["expected"], issues)
    externalSources = inventoryExternalSources(catalog, dataRoot, issues)
    issueCounts = dict(sorted(Counter(issue["code"] for issue in issues).items()))
    errors = sum(issue["severity"] == "error" for issue in issues)
    return {
        "schemaVersion": 1,
        "software": {"packageVersion": __version__, "pythonVersion": platform.python_version()},
        "status": "requiresReview" if errors else "completed",
        "countingUnit": "reviewedFrameRecord",
        "mediaContentHashesComputed": False,
        "configEvidence": catalog.configHashes,
        "reviewRegistryEvidence": registryEvidence,
        "summary": {
            **counts,
            "expected": catalog.data["expected"],
            "errorCount": errors,
            "warningCount": len(issues) - errors,
            "issueCounts": issueCounts,
            "readableVideoCount": sum("byteSize" in video.get("media", {}) for video in videos),
        },
        "canonicalClasses": catalog.labels["canonicalClasses"],
        "labelMaps": catalog.labels["labelMaps"],
        "splitPolicy": catalog.data["splitPolicy"],
        "ownVideos": videos,
        "matches": matches,
        "classCoverage": coverage,
        "externalSources": externalSources,
        "issues": issues,
    }


def summarizeCoverage(matchReports, classNames):
    coverage = []
    for className in classNames:
        support = [match for match in matchReports if match["classFrameCounts"][className]]
        coverage.append(
            {
                "className": className,
                "matchCount": len(support),
                "matchIds": [match["matchId"] for match in support],
                "frameRecordCount": sum(match["classFrameCounts"][className] for match in support),
                "annotationCount": sum(
                    match["classAnnotationCounts"][className] for match in support
                ),
                "threeWayMatchCoveragePossible": len(support) >= 3,
            }
        )
    return coverage


def checkCoverage(counts, coverage, expected, issues):
    for field, expectedValue in expected.items():
        if counts[field] != expectedValue:
            addIssue(
                issues,
                "error",
                "expectedCountMismatch",
                f"{field}: expected {expectedValue}, observed {counts[field]}.",
                field=field,
            )
    for entry in coverage:
        if entry["matchCount"] < 3:
            addIssue(
                issues,
                "warning",
                "limitedMatchCoverage",
                "This class cannot have independent match support in all three splits.",
                canonicalClass=entry["className"],
                matchIds=entry["matchIds"],
            )
    addIssue(
        issues,
        "warning",
        "cutBoundariesPending",
        "Exact internal cut boundaries are pending; no temporal continuity is inferred.",
    )
    addIssue(
        issues,
        "warning",
        "frameUniquenessUnverified",
        "Reviewed frame records are not verified unique decoded images; shared match footage "
        "requires a later overlap audit.",
    )


def inventoryExternalSources(catalog, dataRoot, issues):
    externalSources = []
    for category in ("publicSources", "syntheticSources"):
        for source in catalog.data.get(category, []):
            sourceReport, sourceIssues = inventoryYoloSource(
                source, catalog.labels["labelMaps"][source["labelMapId"]], dataRoot
            )
            sourceReport["origin"] = "public" if category == "publicSources" else "synthetic"
            externalSources.append(sourceReport)
            issues.extend(sourceIssues)
    return externalSources
