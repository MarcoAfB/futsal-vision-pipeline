import copy

import pytest

from futsal.evaluation.modelComparison import validateModelComparison


def testVariantModeOnlyRelaxesDifferentScales():
    first = {"configuration": {"modelSize": "n"},
             "initialWeights": {"modelFamily": "yolo26", "modelSize": "n", "sha256": "nWeights"},
             "stages": [{"taskId": "ballOnly", "results": {"checkpoints": {"best": {"sha256": "best"}}}}]}
    second = copy.deepcopy(first)
    validateModelComparison(first, second, "controlled")
    second["initialWeights"]["sha256"] = "otherWeights"
    for mode in ("controlled", "modelVariants"):
        with pytest.raises(ValueError, match="initialization"):
            validateModelComparison(first, second, mode)
    second["configuration"]["modelSize"] = "s"
    with pytest.raises(ValueError, match="recorded model size"):
        validateModelComparison(first, second, "modelVariants")
    second["initialWeights"]["modelSize"] = "s"
    validateModelComparison(first, second, "modelVariants")
    with pytest.raises(ValueError, match="same model scale"):
        validateModelComparison(first, second, "controlled")
    with pytest.raises(ValueError, match="comparisonMode"):
        validateModelComparison(first, second, "unknown")
