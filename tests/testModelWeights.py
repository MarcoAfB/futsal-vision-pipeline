import pytest

from futsal.modelWeights import prepareWeights, verifyWeights


def testOfficialDownloadsHaveExplicitProvenanceAndCannotBeReplaced(tmp_path, monkeypatch):
    urls = []

    def download(url, path):
        urls.append(url)
        path.write_bytes(b"test checkpoint")

    monkeypatch.setattr("futsal.modelWeights.downloadWeights", download)
    output = tmp_path / "weights"
    metadata = prepareWeights("n", output)
    path = output / "yolo26n.pt"
    assert verifyWeights(path, "n") == metadata
    assert urls == [metadata["sourceUrl"]]
    with pytest.raises(ValueError, match="provenance"):
        verifyWeights(path, "s")
    with pytest.raises(ValueError, match="new directory"):
        prepareWeights("n", output)
    path.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="changed"):
        verifyWeights(path, "n")


def testInterruptedDownloadsDoNotPublishWeights(tmp_path, monkeypatch):
    def interrupted(url, path):
        path.write_bytes(b"partial")
        raise OSError("interrupted download")

    monkeypatch.setattr("futsal.modelWeights.downloadWeights", interrupted)
    output = tmp_path / "weights"
    with pytest.raises(OSError, match="interrupted"):
        prepareWeights("n", output)
    assert not output.exists()
    with pytest.raises(ValueError, match="modelSize"):
        prepareWeights("unknown", output)
