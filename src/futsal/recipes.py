import math
import re
from dataclasses import dataclass
from pathlib import Path

from futsal.catalog import Catalog, loadCatalog, readYaml, requireUnique, resolveSourcePath


@dataclass(slots=True)
class Recipe:
    data: dict
    catalog: Catalog
    evidence: list[dict]


def validateTask(task, canonicalNames, semanticNames, publicIds):
    if not isinstance(task, dict) or not re.fullmatch(r"[a-z][A-Za-z0-9]*", task["taskId"]):
        raise ValueError("taskId must be a camelCase filename-safe identifier.")
    if task["classes"] == "canonical":
        task["classes"] = list(canonicalNames)
    classes = task["classes"]
    if not isinstance(classes, list) or not classes or not all(
        isinstance(name, str) and name for name in classes
    ):
        raise ValueError("Task classes must be a nonempty list of names.")
    requireUnique(classes, "task classes")
    if task["classMap"] == "identity":
        task["classMap"] = {name: name for name in classes}
    mapping = task["classMap"]
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Each task requires an explicit class mapping.")
    if not set(mapping) <= set(canonicalNames) | semanticNames:
        raise ValueError("Task mapping contains unknown source semantics.")
    if not set(mapping.values()) <= set(classes) or set(mapping.values()) != set(classes):
        raise ValueError("Task mapping must cover exactly its declared output classes.")
    sources = task["trainingSources"]
    if not isinstance(sources, list) or not sources or not set(sources) <= publicIds | {"own"}:
        raise ValueError("Task trainingSources must identify registered public sources or own.")
    requireUnique(sources, "task training sources")


def validateRecipe(data, catalog):
    if data["schemaVersion"] != 1:
        raise ValueError("Only curriculum schemaVersion 1 is supported.")
    canonicalNames = [item["name"] for item in catalog.labels["canonicalClasses"]]
    semanticNames = {
        entry["semanticClass"]
        for mapping in catalog.labels["labelMaps"].values()
        for entry in mapping["entries"]
        if entry.get("semanticClass")
    }
    publicIds = {source["sourceId"] for source in catalog.data["publicSources"]}
    tasks = data["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("A curriculum requires at least one task.")
    for task in tasks:
        validateTask(task, canonicalNames, semanticNames, publicIds)
    taskIds = [task["taskId"] for task in tasks]
    requireUnique(taskIds, "task IDs")
    sequence = data["curriculum"]
    if not isinstance(sequence, list) or sequence != taskIds:
        raise ValueError("curriculum must list each task once in its declared order.")
    coverageTask = next((task for task in tasks if task["taskId"] == data["coverageTask"]), None)
    if coverageTask is None or coverageTask["classes"] != canonicalNames:
        raise ValueError("coverageTask must retain the complete ordered canonical ontology.")
    if "own" not in coverageTask["trainingSources"]:
        raise ValueError("coverageTask requires own training data.")
    unsupported = data["unsupportedValidationClasses"]
    if not isinstance(unsupported, list) or not set(unsupported) <= set(canonicalNames):
        raise ValueError("Unknown unsupported validation classes.")
    if data["rareClass"] not in canonicalNames:
        raise ValueError("Unknown rareClass.")
    for value in data["pointBoxes"].values():
        if type(value) not in {float, int} or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("Point box fractions must be finite and in (0, 1].")
    for field in ("widthFraction", "heightFraction"):
        data["pointBoxes"][field]
    validation = data["publicValidation"]
    usedPublic = {source for task in tasks for source in task["trainingSources"]} - {"own"}
    if validation["basis"] != "researcherConfirmation" or not usedPublic <= set(
        validation["sources"]
    ):
        raise ValueError("Public training sources require recorded researcher confirmation.")
    if not {"annotationCompletenessForSourceOntology", "overlapWithOwnRecordings"} <= set(
        validation["checks"]
    ):
        raise ValueError("Public validation must cover annotation scope and overlap.")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", validation["confirmedOn"]):
        raise ValueError("Public validation confirmedOn must be a YYYY-MM-DD string.")
    fractions = catalog.data["splitPolicy"]["targetFractions"]
    if set(fractions) != {"training", "validation", "test"} or any(
        type(value) not in {float, int} or not math.isfinite(value) or not 0 < value < 1
        for value in fractions.values()
    ) or not math.isclose(sum(fractions.values()), 1):
        raise ValueError("Split target fractions must be positive and sum to one.")


def loadRecipe(configPath: Path) -> Recipe:
    configPath = Path(configPath).resolve()
    try:
        data, recipeEvidence = readYaml(configPath)
        catalogPath = resolveSourcePath(configPath.parent, data["catalogPath"])
        catalog = loadCatalog(catalogPath)
        validateRecipe(data, catalog)
        evidence = [recipeEvidence]
        for item in catalog.configHashes:
            evidence.append({**item, "path": (catalogPath.parent / item["path"]).relative_to(
                configPath.parent
            ).as_posix()})
        return Recipe(data, catalog, evidence)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid curriculum configuration: {error}.") from error
