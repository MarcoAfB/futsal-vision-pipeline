import pytest

from futsal.training.memoryRecovery import errorCauses, isMemoryFailure, requireFixedBatch


@pytest.mark.parametrize("message,expected", [
    ("CUDA out of memory. Tried to allocate", True), ("CUDA error: out of memory", True),
    ("CUBLAS_STATUS_ALLOC_FAILED", True), ("CUDNN_STATUS_ALLOC_FAILED", True),
    ("CUDA error: unknown error", False), ("CUDNN_STATUS_INTERNAL_ERROR", False),
    ("unable to find an engine", False), ("out of memory while writing report", False),
])
def testMemoryClassificationRequiresDefiniteAllocationFailure(message, expected):
    assert isMemoryFailure(errorCauses(RuntimeError(message))) is expected


def testNativeGuardPreservesOomCauseAndBlocksBatchOneRetry():
    requireFixedBatch(8, 8, 0)
    for batch, requested, retries in ((4, 8, 1), (1, 1, 1)):
        try:
            try:
                raise RuntimeError("CUDA out of memory")
            except RuntimeError:
                requireFixedBatch(batch, requested, retries)
        except RuntimeError as error:
            assert isMemoryFailure(errorCauses(error))
            assert len(errorCauses(error)) == 2


@pytest.mark.parametrize("batch,valid", [(1, True), (4, True), (0, False), (True, False), (2.5, False), (65, False)])
def testScientificStageBatchUsesIntegerAndNominalBatchValidation(tmp_path, batch, valid):
    from pathlib import Path

    import yaml

    from futsal.datasets.recipes import loadRecipe
    from futsal.training.trainingConfig import loadTrainingConfig, stageParameters

    config = yaml.safe_load(Path("configs/training/yolo26ObjectCurriculum.yaml").read_text())
    config["stages"][1]["parameters"]["batch"] = batch
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(config))
    recipe = loadRecipe("configs/objectCurriculum.yaml").data
    if valid:
        loaded = loadTrainingConfig(path, recipe)
        assert [stageParameters(loaded, stage)["batch"] for stage in loaded["stages"]] == [8, batch, 8]
    else:
        with pytest.raises(ValueError):
            loadTrainingConfig(path, recipe)


def testMemoryRetryAndStrictContinuationAreMutuallyExclusive():
    from futsal.cli import main

    with pytest.raises(SystemExit) as result:
        main(["train", "--datasetDir", "view", "--weights", "weights.pt", "--outputDir", "new",
              "--continueFrom", "previous", "--memoryRetryFrom", "previous"])
    assert result.value.code == 2
