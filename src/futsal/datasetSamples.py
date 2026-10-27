from collections import Counter

from futsal.yoloAnnotations import readYoloAnnotations


def classifySample(sample, task, pointClasses):
    targets = []
    reasons = set(sample.get("readFindings", []))
    if not sample.get("dimensions"):
        reasons.add("dimensionsUnavailable")
    if sample["origin"] == "public" and sample.get("emptyLabel"):
        reasons.add("emptyPublicLabel")
    counts = Counter()
    for index, annotation in enumerate(sample["annotations"]):
        action = annotation["mappingAction"]
        if action == "ignore":
            continue
        if action == "unmapped":
            reasons.add("unmappedLabel")
            continue
        sourceClass = annotation.get("canonicalClass") or annotation.get("semanticClass")
        targetClass = task["classMap"].get(sourceClass)
        if targetClass is None:
            if sourceClass in {"player", "outfieldPlayer", "goalkeeper"}:
                reasons.add("unresolvedAthleteSemantics")
            continue
        reasons.update(annotation["geometryFindings"])
        shape = annotation["shapeType"]
        if sourceClass in pointClasses:
            if shape != "points" or len(annotation["coordinates"]) != 2:
                reasons.add("landmarkShapeMismatch")
        elif shape != "box" and not (sourceClass == "ball" and shape == "ellipse"):
            reasons.add("objectShapeMismatch")
        counts[sourceClass] += 1
        targets.append({"annotationIndex": index, "classId": task["classes"].index(targetClass)})
    for name, count in counts.items():
        if count > 1 and (name == "ball" or name in pointClasses):
            reasons.add("multipleBallAnnotations" if name == "ball" else f"duplicateLandmark:{name}")
    return {"eligible": not reasons, "reasons": sorted(reasons), "targets": targets}


def collectOwnSamples(inventory, recipe):
    pointClasses = {
        item["name"] for item in inventory["canonicalClasses"] if item["geometry"] == "point"
    }
    samples = []
    for video in inventory["ownVideos"]:
        for frame in video["frames"]:
            sample = {
                **frame,
                "sourceId": video["videoId"],
                "origin": "own",
                "mediaPath": video["videoPath"],
                "annotationPath": video["annotationPath"],
                "annotationSha256": video["annotationEvidence"]["sha256"],
            }
            statuses = {}
            for task in recipe.data["tasks"]:
                status = classifySample(sample, task, pointClasses)
                required = set(task["classMap"]) & {
                    item["name"] for item in inventory["canonicalClasses"]
                }
                if video["annotationScope"] != "completeForDeclaredClasses" or not required <= set(
                    video["classesAnnotated"]
                ):
                    status["reasons"].append("annotationScopeUnverified")
                    status["eligible"] = False
                statuses[task["taskId"]] = status
            sample["tasks"] = statuses
            samples.append(sample)
    return samples


def collectPublicSamples(inventory, recipe, dataRoot):
    registered = {item["sourceId"]: item for item in recipe.catalog.data["publicSources"]}
    samples = []
    for report in inventory["externalSources"]:
        sourceId = report["sourceId"]
        tasks = [task for task in recipe.data["tasks"] if sourceId in task["trainingSources"]]
        if not tasks:
            continue
        source = registered[sourceId]
        entries = {
            entry["sourceClassId"]: entry
            for entry in recipe.catalog.labels["labelMaps"][source["labelMapId"]]["entries"]
        }
        for record in report["samples"]:
            issues = []
            document = None
            if record["labelPath"]:
                document = readYoloAnnotations(record["labelPath"], source, entries, dataRoot, issues)
                if document["sha256"] != record["labelSha256"]:
                    raise ValueError(f"Annotation changed during planning: {record['labelPath']}.")
            observations = document["observations"] if document else []
            representedLines = {item["lineNumber"] for item in observations}
            readFindings = {
                issue["code"] for issue in issues if issue.get("lineNumber") not in representedLines
            }
            if document is None:
                readFindings.add("missingOrAmbiguousLabel")
            media = record["media"]
            sample = {
                "recordId": record["recordId"],
                "sourceId": sourceId,
                "origin": "public",
                "matchId": None,
                "frameIndex": None,
                "mediaPath": record["imagePath"],
                "annotationPath": record["labelPath"],
                "annotationSha256": record["labelSha256"],
                "dimensions": [media["width"], media["height"]] if media else None,
                "coordinateSystem": "normalizedCenterBoxes",
                "annotations": observations,
                "emptyLabel": document["empty"] if document else None,
                "readFindings": sorted(readFindings),
            }
            sample["tasks"] = {
                task["taskId"]: classifySample(sample, task, set()) for task in tasks
            }
            samples.append(sample)
    return samples


def collectSamples(inventory, recipe, dataRoot):
    usedSources = {
        source for task in recipe.data["tasks"] for source in task["trainingSources"]
    } | {video["videoId"] for video in inventory["ownVideos"]}
    annotationFindings = {
        "geometryOutOfBounds", "coordinatesOutOfBounds", "degenerateBox", "degenerateGeometry",
        "nonfiniteGeometry", "nonFiniteCoordinates", "invalidCoordinates", "unmappedLabel",
        "annotationFieldCount", "invalidClassId", "unknownClassId", "missingLabel",
        "imageUnreadable", "ambiguousImageStem", "ambiguousLabelStem",
    }
    blockers = [
        issue for issue in inventory["issues"]
        if issue["severity"] == "error" and issue["code"] not in annotationFindings
        and (issue.get("sourceId") is None or issue["sourceId"] in usedSources)
    ]
    if blockers:
        raise ValueError(f"Source inventory blocks preparation: {blockers[0]['code']}.")
    return collectOwnSamples(inventory, recipe) + collectPublicSamples(inventory, recipe, dataRoot)


def eligibleClassCounts(samples, taskId):
    counts = Counter()
    for sample in samples:
        status = sample["tasks"].get(taskId)
        if status and status["eligible"]:
            counts.update(target["classId"] for target in status["targets"])
    return counts


def summarizeEligibility(samples, tasks):
    summary = {}
    for task in tasks:
        sourceCounts = {}
        for sample in samples:
            status = sample["tasks"].get(task["taskId"])
            if status is None:
                continue
            counts = sourceCounts.setdefault(sample["sourceId"], {"eligible": 0, "quarantined": 0})
            counts["eligible" if status["eligible"] else "quarantined"] += 1
        summary[task["taskId"]] = sourceCounts
    return summary
