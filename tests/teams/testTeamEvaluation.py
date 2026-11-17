import numpy as np
import pytest

from futsal.datasets.datasetFiles import readJson
from futsal.evaluation.evaluationData import readJsonLines
from futsal.evaluation.teamRun import evaluateTeams
from futsal.reporting.researchReport import renderReport, verifyEvaluation
from futsal.teams.teamClustering import clusterFeatures


def testFeaturesAndAssignmentRecoverUniformTeamsGoalkeepersAndNames(scene, features, assignmentConfig):
    featureDir, featureState = features
    output = scene["root"] / "teams"
    state = evaluateTeams(featureDir, scene["homography"], scene["reference"], output, assignmentConfig,
                          progress=lambda message: None)
    summary = readJson(output / "evidence" / "summary.json")
    decision = readJson(output / "evidence" / "matches.json")["match01"]
    pipeline, baseline = summary["pipeline"], summary["baseline"]
    report = renderReport(output).read_text(encoding="utf-8")

    assert featureState["cropCount"] == featureState["cleanCropCount"] == 33
    assert state["status"] == "completed" and verifyEvaluation(output)[0]["kind"] == "teamEvaluation"
    assert summary["cohort"]["frames"] == 6 and summary["cohort"]["athleteReferences"] == 33
    assert pipeline["named"]["assignment"]["classAccuracy"] == 1.0 and pipeline["named"]["macroF1"] == 1.0
    assert pipeline["bestSwap"]["swappedMatches"] == []
    assert decision["outfieldNaming"]["source"] == "meanX" and decision["goalkeeperSides"] == ["A", "B"]
    assert baseline["named"]["assignment"]["classAccuracy"] == pytest.approx(12 / 33)
    assert baseline["named"]["assignment"]["roleAccuracy"] == pytest.approx(24 / 33)
    assert len(readJsonLines(output / "evidence" / "assignments.jsonl")) == 33
    assert "# Team and role assignment report" in report and "bestSwap" in report


def testHomographyOfDifferentImagesIsRefused(scene, features, assignmentConfig, foreignHomography):
    with pytest.raises(ValueError, match="disagree on a validation image"):
        evaluateTeams(features[0], foreignHomography, scene["reference"], scene["root"] / "teams", assignmentConfig,
                      progress=lambda message: None)


def testChangedFeatureEvidenceIsRefused(scene, features, assignmentConfig):
    with (features[0] / "evidence" / "crops.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        evaluateTeams(features[0], scene["homography"], scene["reference"], scene["root"] / "teams",
                      assignmentConfig, progress=lambda message: None)


@pytest.mark.teamIntegration
def testUmapReductionKeepsSeparatedUniformGroups():
    generator = np.random.default_rng(0)
    features = np.vstack([centre + generator.normal(0, 0.05, (30, 16)) for centre in np.eye(4, 16) * 5])
    config = {"reduction": "umap", "umapComponents": 3, "umapNeighbors": 15, "umapMinDistance": 0.1,
              "clusterCount": 4, "seed": 42, "minCleanCrops": 20}
    labels, _, _ = clusterFeatures(features, np.ones(len(features), dtype=bool), config)

    assert all(len(set(labels[index * 30:(index + 1) * 30])) == 1 for index in range(4))
    assert len(set(labels)) == 4
