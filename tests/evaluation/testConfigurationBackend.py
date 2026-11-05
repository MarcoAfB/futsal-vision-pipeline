import pytest

from futsal.evaluation.configurationBackend import selectGlobalDetections


def testCanonicalMergeAppliesOneGlobalCapBeforeMatchingAndKeepsTieProvenance():
    objects = [{"classId": 0, "confidence": 0.8, "box": [0, 0, 2, 2]},
               {"classId": 1, "confidence": 0.6, "box": [2, 2, 4, 4]}]
    court = [{"classId": 0, "confidence": 0.9, "box": [3, 3, 4, 4]},
             {"classId": 1, "confidence": 0.8, "box": [1, 1, 2, 2]}]
    combined = selectGlobalDetections([("detailedObjects", objects, {0: 0, 1: 1}),
                                       ("courtPoints", court, {0: 5, 1: 6})], 3)
    assert [item["classId"] for item in combined] == [5, 0, 6]
    assert [item["componentId"] for item in combined] == ["courtPoints", "detailedObjects", "courtPoints"]
    assert objects[0]["classId"] == 0 and court[0]["classId"] == 0


@pytest.mark.trainingIntegration
def testNativeMatchesAreRecomputedAfterTheGlobalCap(tmp_path):
    from futsal.training.trainingBackend import initializeBackend

    initializeBackend(tmp_path / "backend", 2)
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator

    validator = DetectionValidator(save_dir=tmp_path / "native", args={"plots": False})
    truth = {"cls": torch.tensor([0.0]), "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]])}
    values = [{"classId": 0, "confidence": 0.9, "box": [0.0, 0.0, 8.0, 8.0]},
              {"classId": 0, "confidence": 0.8, "box": [0.0, 0.0, 10.0, 10.0]}]

    def native(rows):
        return {"cls": torch.tensor([item["classId"] for item in rows]),
                "bboxes": torch.tensor([item["box"] for item in rows])}

    original = validator._process_batch(native(values), truth)["tp"]
    selected = selectGlobalDetections([("detailedObjects", values, {0: 0})], 1)
    recomputed = validator._process_batch(native(selected), truth)["tp"]
    assert original[0, 0] and not original[1, 0]
    assert recomputed.shape == (1, 10) and recomputed[0, 0]
    assert original[1, 3] and not recomputed[0, 3]
