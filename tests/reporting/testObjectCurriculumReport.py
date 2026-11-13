from futsal.reporting.researchReport import batchLineage


def testBatchLineageDisplaysStageOverridesAndUnknownLegacyEffectiveBatches():
    text = "\n".join(batchLineage({"batchRecovery": "fixed", "configuration": {
        "parameters": {"batch": 8}, "stages": [{"taskId": "genericObjects"},
            {"taskId": "roleObjects", "parameters": {"batch": 2}}, {"taskId": "detailedObjects"}]},
        "stages": [{"taskId": "roleObjects", "results": {"training": {"effectiveBatch": 2}}}]}))
    assert "| genericObjects | 8 | N/A |" in text
    assert "| roleObjects | 2 | 2 |" in text
    assert "| detailedObjects | 8 | N/A |" in text
