import pytest

from futsal.training.trainingExposure import EpochExposure, sourceComposition, verifyNaturalSampling


def testEpochAuditCoversAllOriginsAndLastPartialBatch():
    rows = [{"recordId": "a", "origin": "own", "sourceId": "video", "targets": [{"classId": 0}]},
            {"recordId": "b", "origin": "public", "sourceId": "public", "targets": []},
            {"recordId": "c", "origin": "public", "sourceId": "public", "targets": [{"classId": 0}]}]
    audit = EpochExposure(rows)
    audit.begin(1)
    audit.observe(["c", "a"])
    audit.observe(["b"])
    first = audit.finish()
    assert first["imageCount"] == first["uniqueRecordCount"] == 3
    assert first["coverageVerified"]
    assert first["bySource"] == sourceComposition(rows)
    public = next(row for row in first["bySource"] if row["origin"] == "public")
    assert public["positiveImages"] == public["negativeImages"] == 1
    audit.begin(2)
    audit.observe(["a", "b", "c"])
    assert audit.finish()["orderSha256"] != first["orderSha256"]
    assert audit.completedEpochs == 2


@pytest.mark.parametrize("records", [["a", "a"], ["a"]])
def testEpochAuditRejectsRepeatedOrMissingRecords(records):
    audit = EpochExposure([{"recordId": "a"}, {"recordId": "b"}])
    audit.begin(1)
    audit.observe(records)
    with pytest.raises(ValueError, match="exactly once"):
        audit.finish()
    with pytest.raises(ValueError, match="outside"):
        audit.observe(["unknown"])


@pytest.mark.trainingIntegration
@pytest.mark.parametrize("policy", ["ordered", "replacement", "dropLast", "partialSelection"])
def testNaturalSamplerRefusesOtherExposurePolicies(policy):
    from torch.utils.data import DataLoader, RandomSampler

    rows = list(range(5))
    if policy == "ordered":
        loader = DataLoader(rows, batch_size=2, shuffle=False)
    else:
        sampler = RandomSampler(rows, replacement=policy == "replacement", num_samples=3 if policy == "partialSelection" else 5)
        loader = DataLoader(rows, batch_size=2, sampler=sampler, drop_last=policy == "dropLast")
    with pytest.raises(ValueError, match="Natural mixed"):
        verifyNaturalSampling(loader, 5)
