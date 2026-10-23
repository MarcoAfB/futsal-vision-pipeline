import json
import os
import tempfile
from pathlib import Path


def validateOutputDirectory(outputDir: Path, dataRoot: Path, protectedPaths=()) -> Path:
    outputDir, dataRoot = outputDir.resolve(), dataRoot.resolve()
    if outputDir.is_relative_to(dataRoot):
        raise ValueError("outputDir must be outside dataRoot to protect source data.")
    if outputDir.exists() and not outputDir.is_dir():
        raise ValueError("outputDir must be a directory.")
    protectedPaths = {path.resolve() for path in protectedPaths}
    for filename in ("inventory.json", "report.md"):
        target = outputDir / filename
        if target.is_symlink() or target.resolve().is_relative_to(dataRoot):
            raise ValueError("Output files must not link to source data.")
        if target.resolve() in protectedPaths:
            raise ValueError("Output files must not overwrite input configuration files.")
        if target.exists() and not target.is_file():
            raise ValueError("Output targets must be regular files or absent.")
    return outputDir


def markdownCell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def renderTable(title, headers, rows) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for header in headers) + " |",
    ]
    lines.extend("| " + " | ".join(markdownCell(value) for value in row) + " |" for row in rows)
    return lines + [""]


def renderFindings(inventory) -> list[str]:
    lines = renderTable("Findings", ["Code", "Count"], inventory["summary"]["issueCounts"].items())
    lines += ["The JSON contains all findings; the first 100 are shown here.", ""]
    for issue in inventory["issues"][:100]:
        location = issue.get("sourceId", "catalog")
        if "frameIndex" in issue:
            location += f", frame {issue['frameIndex']}"
        lines.append(
            f"- **{issue['severity']} / {issue['code']}** "
            f"({markdownCell(location)}): {markdownCell(issue['message'])}"
        )
    return lines + [""]


def renderSplitPolicy(policy) -> list[str]:
    fractions = ", ".join(
        f"{name} {value:.0%}" for name, value in policy["targetFractions"].items()
    )
    lines = [
        "## Split preparation",
        "",
        "No split has been assigned. Whole original matches "
        "must remain together, including all cuts and potentially repeated footage.",
        "",
        f"Target fractions for reviewed own frame records: {fractions}.",
        "",
        "Public and synthetic sources are training-only and excluded from that denominator.",
        "",
    ]
    for requirement in policy["rareClassRequirements"]:
        lines.append(
            f"- `{requirement['className']}`: required support in "
            f"{', '.join(requirement['requiredSplits'])}; unsupported in "
            f"{', '.join(requirement['unsupportedSplits'])}. {requirement['reason']}"
        )
    return lines + [
        "",
        "Exact cut boundaries and original-match timestamps remain unresolved. "
        "Video-relative timestamps use reported FPS. No persistent player identity "
        "or temporal continuity between cuts is inferred.",
        "",
    ]


def renderReport(inventory: dict) -> str:
    summary = inventory["summary"]
    lines = [
        "# Dataset inventory",
        "",
        f"Status: **{inventory['status']}**. Schema version: {inventory['schemaVersion']}.",
        "",
        f"Reviewed own frame records: **{summary['reviewedFrames']:,}**. "
        f"Original matches: **{summary['matchCount']}**. Videos: **{summary['videoCount']}**. "
        f"Reviewed intervals: **{summary['reviewedIntervalCount']}**.",
        "",
        f"Findings: {summary['errorCount']} errors and {summary['warningCount']} warnings.",
        "",
        "Counts include source annotations with unresolved findings. They do not certify "
        "correctness, completeness, unique images, or suitability for training.",
        "",
    ]
    lines += renderTable(
        "Reviewed own matches",
        ["Match", "Frame records", "Videos", "Missing classes"],
        [
            (
                item["matchId"],
                item["reviewedFrameRecordCount"],
                ", ".join(item["videoIds"]),
                ", ".join(item["missingClasses"]) or "None",
            )
            for item in inventory["matches"]
        ],
    )
    lines += renderTable(
        "Class coverage in reviewed own data",
        ["Class", "Annotations", "Frame records", "Matches", "Match IDs"],
        [
            (
                item["className"],
                item["annotationCount"],
                item["frameRecordCount"],
                item["matchCount"],
                ", ".join(item["matchIds"]),
            )
            for item in inventory["classCoverage"]
        ],
    )
    lines += renderTable(
        "Public and synthetic sources",
        ["Source", "Origin", "Availability", "Images", "Empty labels"],
        [
            (
                item["sourceId"],
                item["origin"],
                item["availability"],
                item["imageCount"],
                item["emptyLabelCount"],
            )
            for item in inventory["externalSources"]
        ],
    )
    lines += [
        "Generic labels retain source semantics. Missing target annotations and empty "
        "label files do not establish true negatives.",
        "",
    ]
    lines += renderFindings(inventory) + renderSplitPolicy(inventory["splitPolicy"])
    lines += renderTable(
        "Configuration provenance",
        ["Configuration", "SHA-256"],
        [(item["path"], item["sha256"]) for item in inventory["configEvidence"]],
    )
    lines += [
        "Annotation/configuration hashes are SHA-256. Source paths are relative to dataRoot; "
        "configuration paths are relative to the catalog. Media metadata is recorded, "
        "but media content hashes and decoded-frame deduplication were not computed.",
        "",
    ]
    return "\n".join(lines)


def atomicWrite(path: Path, content: str) -> None:
    temporaryPath = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporaryPath = Path(stream.name)
            stream.write(content)
        os.replace(temporaryPath, path)
    finally:
        if temporaryPath is not None and temporaryPath.exists():
            temporaryPath.unlink()


def writeInventory(
    inventory: dict, outputDir: Path, dataRoot: Path, protectedPaths=()
) -> tuple[Path, Path]:
    outputDir = validateOutputDirectory(outputDir, dataRoot, protectedPaths)
    jsonContent = (
        json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )
    markdownContent = renderReport(inventory)
    outputDir.mkdir(parents=True, exist_ok=True)
    inventoryPath, reportPath = outputDir / "inventory.json", outputDir / "report.md"
    atomicWrite(inventoryPath, jsonContent)
    atomicWrite(reportPath, markdownContent)
    return inventoryPath, reportPath
