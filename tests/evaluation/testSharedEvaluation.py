from pathlib import Path

import pytest

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.datasets.mediaRegistration import registerMedia
from futsal.datasets.sharedEquivalence import verifyRunView
from futsal.datasets.sharedViews import prepareView
from futsal.evaluation.configurationRun import evaluateConfiguration
from futsal.inventory.catalog import hashFile
from futsal.training.trainingData import selectTrainingData


def testEquivalentViewPreservesArchivedSelectionAndRejectsChangedTargets(trainingBundle, preparedSources, tmp_path):
    bundle, _, config = trainingBundle
    store, view = tmp_path / "store", tmp_path / "view"
    registerMedia(store, bundle)
    prepareView(store, preparedSources[1], view)
    runDir = tmp_path / "run"
    runDir.mkdir()
    selection = selectTrainingData(bundle, config)
    writeJson(runDir / "selection.json", selection)
    run = {"configuration": config, "datasetDirectory": str(bundle),
           "contractId": readJson(bundle / "contract.json")["contractId"],
           "selectionSha256": hashFile(runDir / "selection.json")}
    equivalence = verifyRunView(runDir, view, run)
    assert equivalence["trainingContractId"] == run["contractId"]
    assert equivalence["evaluationContractId"] == readJson(view / "contract.json")["contractId"]
    selection["tasks"]["genericObjects"]["training"][0]["targets"][0]["classId"] = 1
    writeJson(runDir / "selection.json", selection)
    run["selectionSha256"] = hashFile(runDir / "selection.json")
    with pytest.raises(ValueError, match="changed ordered records"):
        verifyRunView(runDir, view, run)


def testJointEvaluationReusesSharedAssetsForLegacyComponents(configurationEvaluation, projectionBundle, tmp_path):
    _, runs, data, parent, _ = configurationEvaluation
    store, view = tmp_path / "store", tmp_path / "view"
    registerMedia(store, parent)
    registerMedia(store, data)
    prepareView(store, projectionBundle[2], view)
    output = tmp_path / "sharedJoint"
    result = evaluateConfiguration(*runs, view, parent, "configs/evaluation/researchValidation.yaml", output,
                                   workRoot=tmp_path / "sharedWork", mediaStoreDir=store)
    assert result["status"] == "completed", result.get("error")
    assert all(item["datasetEquivalence"] for item in result["components"])
    runtime = Path(result["workDirectory"]) / "runtimeData"
    assert not list(runtime.rglob("*.png")) and not list(runtime.rglob("*.jpg"))
    assert result["runtimeAfter"]
