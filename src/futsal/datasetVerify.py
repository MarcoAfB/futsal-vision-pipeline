from collections import Counter, defaultdict
from pathlib import Path

from futsal.catalog import readYaml, requireUnique, resolveSourcePath
from futsal.datasetExport import exportSelections, sampleToken, validateExportSummary
from futsal.datasetFiles import hashJson, readJson, verifyEvidence
from futsal.datasetPlan import loadDatasetPlan
from futsal.detectionGeometry import convertTargets, renderYoloLabels, validNormalizedBox
from futsal.mediaMetadata import readImageMetadata
from futsal.recipes import activeTasks
from futsal.yoloInventory import scanFiles


def verifyTaskConfigurations(root, tasks):
    paths = set()
    for task in tasks:
        relativePath = f"{task['taskId']}/dataset.yaml"
        config, _ = readYaml(resolveSourcePath(root, relativePath))
        expected = {"train": "images/train", "val": "images/val",
                    "names": dict(enumerate(task["classes"]))}
        if config != expected:
            raise ValueError(f"Dataset YAML differs from the portable development contract: {relativePath}.")
        paths.add(relativePath)
    return paths


def verifyExportedRow(root, row, sample, task, pointBoxes, artifactHashes, sourceHashes):
    for field in ("sourceId", "origin", "matchId", "frameIndex", "mediaPath", "annotationPath",
                  "annotationSha256", "dimensions"):
        if row[field] != sample[field]:
            raise ValueError(f"Exported provenance differs from plan: {row['recordId']} / {field}.")
    splitDirectory = "train" if row["split"] == "training" else "val"
    suffix = Path(sample["mediaPath"]).suffix.lower() if sample["origin"] == "public" else ".png"
    stem = f"{task['taskId']}/images/{splitDirectory}/{sampleToken(sample)}"
    imagePath = stem + suffix
    labelPath = stem.replace("/images/", "/labels/") + ".txt"
    if row["imagePath"] != imagePath or row["labelPath"] != labelPath:
        raise ValueError("Exported paths differ from planned task and split membership.")
    if row["imageSha256"] != artifactHashes[imagePath] or row["labelSha256"] != artifactHashes[labelPath]:
        raise ValueError("Manifest hashes differ from artifact evidence.")
    if sample["origin"] == "public" and row["imageSha256"] != sourceHashes[sample["mediaPath"]]:
        raise ValueError("Public image bytes differ from their source snapshot.")
    expectedTargets = convertTargets(sample, task, pointBoxes)
    if row["targets"] != expectedTargets:
        raise ValueError("Derived targets differ from native annotation references.")
    labelText = resolveSourcePath(root, labelPath).read_text(encoding="utf-8")
    if labelText != renderYoloLabels(expectedTargets):
        raise ValueError("Exported labels differ from native annotation conversion.")
    counts = Counter()
    for line in labelText.splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Detection labels require five fields: {labelPath}.")
        classId = int(fields[0])
        if not 0 <= classId < len(task["classes"]) or not validNormalizedBox(
            [float(value) for value in fields[1:]]
        ):
            raise ValueError(f"Invalid class or normalized box: {labelPath}.")
        counts[classId] += 1
    media = readImageMetadata(resolveSourcePath(root, imagePath))
    if [media["width"], media["height"]] != sample["dimensions"]:
        raise ValueError(f"Exported image dimensions differ from annotations: {imagePath}.")
    return {imagePath, labelPath}, counts


def verifyReviews(manifest, plan, samples):
    expected = {}
    for sample in samples.values():
        quarantined = {task["taskId"]: sample["tasks"][task["taskId"]]["reasons"]
                       for task, _ in exportSelections(plan, sample)
                       if not sample["tasks"][task["taskId"]]["eligible"]}
        if quarantined:
            expected[sample["recordId"]] = quarantined
    reviews = manifest["reviews"]
    requireUnique([review["recordId"] for review in reviews], "review records")
    if set(expected) != {review["recordId"] for review in reviews}:
        raise ValueError("Review membership includes reserved test or unplanned samples.")
    paths = set()
    for review in reviews:
        if review["quarantinedTasks"] != expected[review["recordId"]]:
            raise ValueError("Review reasons differ from planned quarantine.")
        sample = samples[review["recordId"]]
        if review["imagePath"] is None:
            if sample["origin"] != "public" or sample["dimensions"] is not None:
                raise ValueError("A decodable review sample must have a preview.")
        else:
            path = f"review/{sampleToken(sample)}.jpg"
            if review["imagePath"] != path:
                raise ValueError("Review preview path differs from its source record.")
            paths.add(path)
    return paths


def verifyDataset(datasetDir, progress=print):
    root = Path(datasetDir).resolve()
    try:
        contract = readJson(root / "contract.json")
        if contract["schemaVersion"] != 1 or contract["kind"] != "preparedCurriculumDataset":
            raise ValueError("Unsupported prepared dataset contract.")
        if contract["contractId"] != hashJson({key: value for key, value in contract.items()
                                               if key != "contractId"}):
            raise ValueError("Dataset contract checksum mismatch.")
        if contract["testExported"] is not False or contract["sourcePreservationVerified"] is not True:
            raise ValueError("Dataset contract must reserve test data and verify source preservation.")
        artifacts = contract["artifactEvidence"]
        requireUnique([item["path"] for item in artifacts], "artifact paths")
        artifactHashes = {item["path"]: item["sha256"] for item in artifacts}
        observed = {path.relative_to(root).as_posix() for path in scanFiles(root)} - {"contract.json"}
        if observed != set(artifactHashes):
            raise ValueError("Dataset has missing or unexpected files.")
        progress("Checking prepared artifact hashes...")
        verifyEvidence(root, artifacts)
        plan = loadDatasetPlan(root / "plan.json")
        manifest = readJson(root / "manifest.json")
        if manifest["schemaVersion"] != 1 or not contract["planId"] == manifest["planId"] == plan["planId"]:
            raise ValueError("Plan, manifest and dataset contract disagree.")
        summary = validateExportSummary(plan, manifest["samples"])
        if summary != manifest["summary"] or summary != contract["summary"]:
            raise ValueError("Exported class coverage differs from the declared summary.")
        tasks = {task["taskId"]: task for task in activeTasks(plan["recipe"])}
        allowedPaths = {"plan.json", "manifest.json", "review.md", "report.md"}
        allowedPaths.update(verifyTaskConfigurations(root, tasks.values()))
        samples = {sample["recordId"]: sample for sample in plan["samples"]}
        sourceHashes = {item["path"]: item["sha256"] for item in plan["sourceEvidence"]}
        labelCounts = defaultdict(Counter)
        rows = manifest["samples"]
        for index, row in enumerate(rows):
            if index % 1000 == 0:
                progress(f"Checking exported images and labels: {index}/{len(rows)}")
            paths, counts = verifyExportedRow(root, row, samples[row["recordId"]], tasks[row["taskId"]],
                                              plan["recipe"]["pointBoxes"], artifactHashes, sourceHashes)
            allowedPaths.update(paths)
            labelCounts[(row["taskId"], row["split"])].update(counts)
        for taskId, splits in summary.items():
            for split, counts in splits.items():
                actual = labelCounts[(taskId, split)]
                expected = {name: actual[index] for index, name in enumerate(tasks[taskId]["classes"])}
                if counts["classAnnotationCounts"] != expected:
                    raise ValueError("Label files do not reproduce the declared per-class support.")
        allowedPaths.update(verifyReviews(manifest, plan, samples))
        if allowedPaths != observed:
            raise ValueError("Unexpected artifacts violate the development-only export contract.")
        return {"contractId": contract["contractId"], "planId": plan["planId"],
                "verifiedArtifactCount": len(artifacts), "exportedImageCount": len(rows),
                "reservedTestRecordCount": sum(sample["origin"] == "own"
                    and plan["frozenSplit"]["assignments"][sample["matchId"]] == "test"
                    for sample in samples.values()), "summary": summary}
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"Invalid prepared dataset contract: {error}.") from error
