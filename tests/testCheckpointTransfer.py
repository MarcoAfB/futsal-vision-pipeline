import pytest

from futsal.checkpointTransfer import classificationKeys, loadAudited


@pytest.mark.trainingIntegration
def testNativeTransferUsesNamesAcrossBothBranches(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "settings"))
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    torch = pytest.importorskip("torch")
    from ultralytics.nn.tasks import DetectionModel

    torch.set_num_threads(2)
    source = DetectionModel("yolo26n.yaml", nc=2, verbose=False)
    source.names = {0: "player", 1: "ball"}
    with torch.no_grad():
        for key in classificationKeys(source):
            tensor = source.state_dict()[key]
            tensor[0].fill_(0.123)
            tensor[1].fill_(0.456)
    target = DetectionModel("yolo26n.yaml", nc=3, verbose=False)
    target.names = {0: "ball", 1: "outfieldPlayer", 2: "goalkeeper"}
    audit = loadAudited(target, source)
    assert audit["matchedClasses"] == ["ball"]
    assert len(audit["verifiedRows"]) == 12
    assert {row["sourceIndex"] for row in audit["verifiedRows"]} == {1}
    assert {row["targetIndex"] for row in audit["verifiedRows"]} == {0}
    assert any("one2one_cv3" in row["tensor"] for row in audit["verifiedRows"])
    assert audit["newClasses"] == ["outfieldPlayer", "goalkeeper"]
    final = DetectionModel("yolo26n.yaml", nc=25, verbose=False)
    final.names = {index: ("ball" if index == 4 else f"landmark{index}") for index in range(25)}
    finalAudit = loadAudited(final, target)
    assert {row["targetIndex"] for row in finalAudit["verifiedRows"]} == {4}
    assert finalAudit["verifiedCompatibleTensorCount"] > 400
