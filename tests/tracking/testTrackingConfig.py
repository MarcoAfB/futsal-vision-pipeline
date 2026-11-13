from pathlib import Path

import pytest
import yaml

from futsal.tracking.trackingConfig import loadTrackingConfig, loadTrackingRuns


def testTrackingDefaultsAndInvalidProtocols(tmp_path):
    source = Path("configs/evaluation/ballTracking.yaml")
    config = loadTrackingConfig(source)
    assert len(config["trackers"]) == 6 and config["imgsz"] == 960
    for key, value in (("half", True), ("batch", 2), ("imgsz", 33), ("seed", True),
                       ("detectorConfidence", 0.9), ("trackers", ["bytetrack", "bytetrack"]),
                       ("operationalIou", float("nan"))):
        path = tmp_path / "protocol.yaml"
        path.write_text(yaml.safe_dump({**config, key: value}))
        with pytest.raises(ValueError):
            loadTrackingConfig(path)


def testTrackingRunPathsAreLocalOrderedAndDistinct(tmp_path):
    path = tmp_path / "runs.yaml"
    path.write_text(yaml.safe_dump({"schemaVersion": 1, "runs": {"x": "runX", "n": "runN"}}))
    runs = loadTrackingRuns(path)
    assert list(runs) == ["n", "x"] and runs["n"] == tmp_path / "runN"
    path.write_text(yaml.safe_dump({"schemaVersion": 1, "runs": {"n": "same", "s": "same"}}))
    with pytest.raises(ValueError, match="distinct"):
        loadTrackingRuns(path)
