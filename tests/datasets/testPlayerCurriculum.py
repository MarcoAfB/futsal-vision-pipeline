import copy
from pathlib import Path

import pytest

from futsal.datasets.datasetFiles import hashJson, readJson, writeJson
from futsal.datasets.datasetPlan import loadDatasetPlan
from futsal.datasets.playerCurriculum import buildPlayerCurriculumPlan
from futsal.datasets.projectionPlan import buildProjectionPlan
from futsal.datasets.recipes import activeTasks, loadRecipe
from futsal.datasets.sharedViews import verifyView
from futsal.training.trainingConfig import loadTrainingConfig, resolveTrainingConfig
from futsal.training.trainingData import selectTrainingData


def testPlayerViewKeepsNativeEvidenceAndUsesOnlyAthleteTargets(playerCurriculumView, trainingBundle, preparedSources):
    view, store, profile = playerCurriculumView
    parent = loadDatasetPlan(trainingBundle[0] / "plan.json")
    plan = loadDatasetPlan(view / "plan.json")
    assert plan["parentPlanId"] == parent["planId"]
    for key in ("catalog", "sourceEvidence", "frozenSplit"):
        assert plan[key] == parent[key]
    for before, after in zip(parent["samples"], plan["samples"], strict=True):
        assert {key: value for key, value in before.items() if key != "tasks"} == {
            key: value for key, value in after.items() if key != "tasks"}
        assert all(after["tasks"][key] == value for key, value in before["tasks"].items())
    tasks = activeTasks(plan["recipe"])
    assert [len(task["classes"]) for task in tasks] == [1, 2, 4]
    assert "unified25" not in plan["recipe"]["curriculum"]
    rows = readJson(view / "manifest.json")["samples"]
    assert not any(row["split"] == "test" for row in rows)
    assert not list(view.rglob("*.png")) and not list(view.rglob("*.jpg"))
    for row in rows:
        task = next(task for task in tasks if task["taskId"] == row["taskId"])
        assert "ball" not in task["classes"] and "ball" not in task["classMap"]
        for target in row["targets"]:
            assert 0 <= target["classId"] < len(task["classes"])
        if row["split"] == "training":
            assert row["origin"] == ("own" if row["taskId"] == "detailedPlayers" else "public")
        else:
            assert row["origin"] == "own"
    validations = [{row["recordId"] for row in rows if row["taskId"] == task["taskId"] and row["split"] == "validation"} for task in tasks]
    assert validations[0] == validations[1] == validations[2]
    assert verifyView(view, store)["contractId"]
    for size in ("n", "s", "m", "l", "x"):
        config = resolveTrainingConfig(loadTrainingConfig(profile, plan["recipe"]), modelSize=size)
        selection = selectTrainingData(view, config)
        assert all(all(value > 0 for value in task["trainingSupport"].values()) for task in selection["tasks"].values())
    with pytest.raises(ValueError, match="own-data"):
        buildProjectionPlan(parent, loadRecipe(preparedSources[1].with_name("playerCurriculum.yaml")))


@pytest.mark.parametrize("change", ["ball", "order", "publicSource", "classMap", "coverage", "catalog"])
def testPlayerDerivationRejectsIncompatibleContracts(playerCurriculumView, trainingBundle, preparedSources, change):
    parent = loadDatasetPlan(trainingBundle[0] / "plan.json")
    original = copy.deepcopy(parent)
    recipe = loadRecipe(preparedSources[1].with_name("playerCurriculum.yaml"))
    if change == "ball":
        recipe.data["tasks"][0]["classes"].append("ball")
        recipe.data["tasks"][0]["classMap"]["ball"] = "ball"
    elif change == "order":
        recipe.data["curriculum"].reverse()
    elif change == "publicSource":
        recipe.data["tasks"][0]["trainingSources"] = ["own"]
    elif change == "classMap":
        recipe.data["tasks"][1]["classMap"]["goalkeeperTeamA"] = "outfieldPlayer"
    elif change == "coverage":
        recipe.data["pointBoxes"]["widthFraction"] *= 2
    else:
        recipe.catalog.data["expected"]["reviewedFrames"] += 1
    with pytest.raises(ValueError):
        buildPlayerCurriculumPlan(parent, recipe)
    assert parent == original


def testPlayerViewRecoversBallOnlyFailuresAndBindsMissingAssets(preparedSources, tmp_path):
    import sys
    from xml.etree import ElementTree

    import cv2
    import numpy as np
    import yaml

    from futsal.datasets.datasetExport import prepareDataset
    from futsal.datasets.datasetPlan import buildDatasetPlan, writeDatasetPlan
    from futsal.datasets.mediaRegistration import registerMedia
    from futsal.datasets.mediaStore import readBindings, resolveAsset, storeCollections
    from futsal.datasets.objectCurriculum import prepareObjectCurriculum
    from futsal.datasets.sharedCli import mediaInventory

    dataRoot, sourceConfig, _ = preparedSources
    for index, name in enumerate(("futsal4400", "futsalMot")):
        labels = dataRoot / name / "image.txt"
        content = labels.read_text()
        labels.write_text(content.replace("0.1 0.1", "0 0.1", 1))
        (labels.parent / "additional.txt").write_text(content)
        assert cv2.imwrite(str(labels.parent / "additional.jpg"), np.full((64, 96, 3), 180 + index * 20, dtype=np.uint8))
    for path in dataRoot.glob("video*.xml"):
        tree = ElementTree.parse(path)
        tree.getroot().find("track[@label='Bola']/ellipse").set("rx", "0")
        tree.write(path, encoding="utf-8")
    parent = buildDatasetPlan(sourceConfig, dataRoot)
    native = writeDatasetPlan(parent, tmp_path / "native", dataRoot)
    bundle = tmp_path / "bundle"
    prepareDataset(native, dataRoot, bundle, progress=lambda message: None)
    recipe = yaml.safe_load(Path("configs/playerCurriculum.yaml").read_text())
    recipe.pop("splitPath")
    for index in (0, 1):
        recipe["tasks"][index]["trainingSources"] = parent["recipe"]["tasks"][index]["trainingSources"]
    recipePath = sourceConfig.with_name("playerCurriculum.yaml")
    recipePath.write_text(yaml.safe_dump(recipe))
    store, view = tmp_path / "store", tmp_path / "view"
    registerMedia(store, bundle)
    prepareObjectCurriculum(bundle, recipePath, store, tmp_path / "derived", view,
                            curriculum="players", dataRoot=dataRoot, progress=lambda message: None)
    rows = readJson(view / "manifest.json")["samples"]
    assert sum(row["taskId"] == "genericPlayers" and row["split"] == "training" for row in rows) == 2
    assert sum(row["taskId"] == "rolePlayers" and row["split"] == "training" for row in rows) == 2
    assert sum(row["taskId"] == "detailedPlayers" and row["split"] == "training" for row in rows) == 2
    assert all(sum(row["taskId"] == task and row["split"] == "validation" for row in rows) == 2
               for task in ("genericPlayers", "rolePlayers", "detailedPlayers"))
    assert not list(view.rglob("*.png")) and not list(view.rglob("*.jpg"))
    assert verifyView(view, store)["contractId"]
    registered = {record for collection in storeCollections(store) for record in collection["images"]}
    reserved = {sample["recordId"] for sample in parent["samples"] if sample["origin"] == "own" and
                parent["frozenSplit"]["assignments"][sample["matchId"]] == "test"}
    assert not registered & reserved
    bindings = readBindings(store)
    additional = next(row for row in rows if row["origin"] == "public" and row["mediaPath"].endswith("image.jpg"))
    assert resolveAsset(store, additional["assetId"], bindings=bindings).is_relative_to(dataRoot)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import sharedDataTransfer

    receiver = tmp_path / "receiver"
    registerMedia(receiver, bundle)
    inventory = mediaInventory(receiver)
    inventoryPath = tmp_path / "receiverInventory.json"
    writeJson(inventoryPath, inventory)
    archive, receipt = tmp_path / "playerView.tar.gz", tmp_path / "playerView.receipt.json"
    transfer = sharedDataTransfer.packView(view, store, inventoryPath, archive, receipt)
    expected = readJson(view / "contract.json")["assets"].keys() - inventory["assets"].keys()
    assert expected and transfer["metadata"]["transferred"].keys() == expected
    received = tmp_path / "received"
    sharedDataTransfer.receiveView(receiver, archive, receipt, received)
    assert readJson(received / "view/contract.json") == readJson(view / "contract.json")
    assert verifyView(received / "view", receiver)["contractId"]
    assert not list((received / "view").rglob("*.png"))
    writeJson(inventoryPath, mediaInventory(receiver))
    second = sharedDataTransfer.packView(view, store, inventoryPath, tmp_path / "metadataOnly.tar.gz", tmp_path / "metadataOnly.json")
    assert second["metadata"]["transferred"] == {}


@pytest.mark.parametrize("failure", ["playerGeometry", "unmapped", "unreadable"])
def testPlayerEligibilityRetainsRelevantQuarantine(playerCurriculumView, trainingBundle, preparedSources, failure):
    parent = loadDatasetPlan(trainingBundle[0] / "plan.json")
    recipe = loadRecipe(preparedSources[1].with_name("playerCurriculum.yaml"))
    sample = next(sample for sample in parent["samples"] if sample["sourceId"] == "futsal4400")
    athlete = next(item for item in sample["annotations"] if item.get("semanticClass") == "player")
    if failure == "playerGeometry":
        athlete["geometryFindings"] = ["degenerateBox"]
    elif failure == "unmapped":
        athlete["mappingAction"] = "unmapped"
    else:
        sample["dimensions"] = None
    parent["planId"] = hashJson({key: value for key, value in parent.items() if key != "planId"})
    derived = buildPlayerCurriculumPlan(parent, recipe)
    actual = next(item for item in derived["samples"] if item["recordId"] == sample["recordId"])
    assert not actual["tasks"]["genericPlayers"]["eligible"]
