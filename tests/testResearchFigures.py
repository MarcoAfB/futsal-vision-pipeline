from pathlib import Path

import pytest

from futsal.evaluationData import loadEvaluationConfig
from futsal.researchFigures import stageFigures


@pytest.mark.reportingIntegration
def testFiguresExposeUnsupportedClassesAndDoNotInventLearningCurves(tmp_path):
    detection = {"perClass": [{"className": "ball", "ap50To95": 0.00001, "support": 1},
                               {"className": "unsupported", "ap50To95": None, "support": 0}],
                 "byMatch": {"match": {"imageCount": 1, "groups": {"all": {"ap50To95": 0.00001}}}},
                 "prCurves": {}, "confusionLabels": ["ball", "unsupported", "background"],
                 "confusionMatrix": [[0, 0, 1], [0, 0, 0], [0, 0, 0]]}
    points = {"perClass": [], "support": 0}
    history = tmp_path / "results.csv"
    history.write_text("epoch,train/box_loss,metrics/mAP50-95(B)\n1,2.0,0.00001\n")
    config = loadEvaluationConfig(Path("configs/evaluation/researchValidation.yaml"))
    directory = tmp_path / "figures"
    figures = stageFigures(detection, points, config, [], history, directory)
    assert set(figures) == {"classes", "matches"}
    svg = (directory / "classDetection.svg").read_text(encoding="utf-8")
    assert "N/A: no reference" in svg and "zoomed scale" in svg
    assert (directory / "classDetection.png").stat().st_size > 1000
    assert not (directory / "trainingHistory.png").exists()
