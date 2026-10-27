import platform
from pathlib import Path

import cv2
import yaml

from futsal import __version__
from futsal.catalog import Catalog, readYaml, requireUnique, resolveSourcePath
from futsal.datasetFiles import (
    hashJson,
    readJson,
    sourceEvidence,
    stagedDirectory,
    verifyEvidence,
    writeJson,
)
from futsal.datasetSamples import collectSamples, summarizeEligibility
from futsal.inventory import buildInventory
from futsal.matchSplits import (
    frozenSplitDocument,
    selectAssignments,
    summarizeSplits,
    validateFrozenSplit,
)
from futsal.recipes import Recipe, loadRecipe, validateRecipe
from futsal.report import renderTable


def planRecipe(plan):
    catalog = Catalog(plan["catalog"]["data"], plan["catalog"]["labels"], [])
    recipe = Recipe(plan["recipe"], catalog, plan["configEvidence"])
    validateRecipe(recipe.data, catalog)
    return recipe


def validatePlan(plan):
    try:
        if plan["schemaVersion"] != 1 or plan["kind"] != "curriculumDatasetPlan":
            raise ValueError("Unsupported dataset plan schema.")
        if plan["planId"] != hashJson({key: value for key, value in plan.items() if key != "planId"}):
            raise ValueError("Dataset plan checksum mismatch.")
        recipe = planRecipe(plan)
        requireUnique([sample["recordId"] for sample in plan["samples"]], "sample record IDs")
        ownCount = sum(sample["origin"] == "own" for sample in plan["samples"])
        if ownCount != recipe.catalog.data["expected"]["reviewedFrames"]:
            raise ValueError("Plan changed the authoritative reviewed-frame denominator.")
        validateFrozenSplit(plan["frozenSplit"], plan["samples"], recipe)
        return recipe
    except (KeyError, TypeError, StopIteration) as error:
        raise ValueError(f"Invalid dataset plan: {error}.") from error


def buildDatasetPlan(configPath, dataRoot):
    configPath, dataRoot = Path(configPath).resolve(), Path(dataRoot).resolve()
    recipe = loadRecipe(configPath)
    inventory = buildInventory(dataRoot, recipe.catalog)
    samples = collectSamples(inventory, recipe, dataRoot)
    evidence = list(recipe.evidence)
    if recipe.data.get("splitPath"):
        splitPath = resolveSourcePath(configPath.parent, recipe.data["splitPath"])
        split, splitEvidence = readYaml(splitPath)
        validateFrozenSplit(split, samples, recipe)
        evidence.append({**splitEvidence, "path": recipe.data["splitPath"]})
    else:
        split = frozenSplitDocument(samples, selectAssignments(samples, recipe), recipe)
    plan = {
        "schemaVersion": 1,
        "kind": "curriculumDatasetPlan",
        "software": {"packageVersion": __version__, "pythonVersion": platform.python_version(),
                     "opencvVersion": cv2.__version__},
        "recipe": recipe.data,
        "catalog": {"data": recipe.catalog.data, "labels": recipe.catalog.labels},
        "configEvidence": evidence,
        "sourceEvidence": sourceEvidence(inventory, samples, dataRoot),
        "inventorySummary": inventory["summary"],
        "frozenSplit": split,
        "eligibilitySummary": summarizeEligibility(samples, recipe.data["tasks"]),
        "ownSplitSummaries": {
            task["taskId"]: summarizeSplits(samples, split["assignments"], task)
            for task in recipe.data["tasks"]
        },
        "samples": samples,
    }
    verifyEvidence(configPath.parent, evidence)
    plan["planId"] = hashJson(plan)
    validatePlan(plan)
    return plan


def renderPlanReport(plan):
    lines = ["# Curriculum dataset plan", "", f"Plan ID: `{plan['planId']}`.", "",
             "Test images and labels are reserved; only metadata is present in this plan.", ""]
    for task in plan["recipe"]["tasks"]:
        taskId = task["taskId"]
        summary = plan["ownSplitSummaries"][taskId]
        lines += renderTable(f"{taskId}: own split projection", [
            "Split", "Reviewed records", "Reviewed fraction", "Eligible", "Retained fraction",
            "Missing eligible classes",
        ], [(split, item["reviewedFrameCount"], f"{item['reviewedFraction']:.2%}",
             item["eligibleFrameCount"], f"{item['eligibleFraction']:.2%}",
             ", ".join(item["missingClasses"]) or "None") for split, item in summary.items()])
        lines += renderTable(f"{taskId}: source eligibility", ["Source", "Eligible", "Quarantined"],
                             [(source, counts["eligible"], counts["quarantined"])
                              for source, counts in plan["eligibilitySummary"][taskId].items()])
    lines += ["Reviewed fractions use all reviewed own frame records. Retained fractions use",
              "all eligible own records for that task, including reserved test records.",
              "Own training projections for public-only stages are not exported as training data.",
              "Public validation and overlap review are researcher declarations. No automated",
              "overlap search or deduplication is performed. Internal cuts and athlete identities",
              "remain uncertified. Training and final test export are subsequent work.", ""]
    return "\n".join(lines)


def writeDatasetPlan(plan, outputDir, dataRoot, protectedPaths=()):
    validatePlan(plan)
    with stagedDirectory(outputDir, dataRoot, protectedPaths) as staging:
        writeJson(staging / "plan.json", plan)
        (staging / "splits.yaml").write_text(yaml.safe_dump(plan["frozenSplit"], sort_keys=False),
                                             encoding="utf-8", newline="\n")
        (staging / "report.md").write_text(renderPlanReport(plan), encoding="utf-8", newline="\n")
    return Path(outputDir) / "plan.json"


def loadDatasetPlan(path):
    plan = readJson(path)
    validatePlan(plan)
    return plan
