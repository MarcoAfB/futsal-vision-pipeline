import copy
from pathlib import Path

from futsal.datasets.datasetFiles import verifyEvidence
from futsal.datasets.rawSources import verifyRawSources
from futsal.datasets.recipes import Recipe
from futsal.inventory.catalog import Catalog


def scientificCatalog(data):
    result = copy.deepcopy(data)
    for field in ("schemaVersion", "legacyReviewPath", "reviewPolicy", "syntheticSources"):
        result.pop(field, None)
    for video in result["ownVideos"]:
        for field in ("videoPath", "annotationPath", "legacyMatchId"):
            video.pop(field, None)
        for interval in video["reviewedIntervals"]:
            provenance = interval["provenance"]
            if provenance.get("reviewedRangeBasis") in {"legacyReviewRegistry", "catalogReviewedIntervals"}:
                provenance["reviewedRangeBasis"] = "registeredReviewedIntervals"
    for source in result["publicSources"]:
        source["recordNamespace"] = source.get("recordNamespace", source["imageRoot"])
        source["metadataFiles"] = [Path(path).relative_to(source["imageRoot"]).as_posix() for path in source["metadataFiles"]]
        for field in ("imageRoot", "labelRoot", "labelMapEvidence"):
            source.pop(field, None)
    return result


def projectionRecipe(parent, recipe, dataRoot):
    registered = {"data": recipe.catalog.data, "labels": recipe.catalog.labels}
    if registered == parent["catalog"]:
        return recipe
    previous = parent["catalog"]
    current = recipe.catalog
    if previous["data"]["schemaVersion"] != 1 or current.data["schemaVersion"] != 2 or (
        previous["labels"] != current.labels or scientificCatalog(previous["data"]) != scientificCatalog(current.data)
    ):
        raise ValueError("Source relocation changed the registered scientific catalogue.")
    manifest = verifyRawSources(dataRoot)
    if manifest["catalog"] != current.data:
        raise ValueError("Raw source contract differs from the current catalogue.")
    aliases = manifest["sourceAliases"]
    byId = {video["videoId"]: video for video in current.data["ownVideos"]}
    for video in previous["data"]["ownVideos"]:
        for field in ("videoPath", "annotationPath"):
            if aliases.get(video[field]) != byId[video["videoId"]][field]:
                raise ValueError("Source relocation changed the recording identity.")
    metadataOnly = {previous["data"]["legacyReviewPath"],
                    *(source["labelMapEvidence"] for source in previous["data"]["publicSources"])}
    evidence = [item for item in parent["sourceEvidence"] if item["path"] not in metadataOnly]
    if not {item["path"] for item in evidence} <= set(aliases):
        raise ValueError("Source relocation omitted previously registered raw inputs.")
    verifyEvidence(dataRoot, evidence)
    return Recipe(recipe.data, Catalog(previous["data"], previous["labels"], []), recipe.evidence)
