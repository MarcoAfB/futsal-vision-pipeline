from pathlib import Path

import pytest
import yaml

from futsal.tracking.trackingConfig import loadTrackingConfig, loadTrackingRuns, resolveTrackedRuns


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


def testTrackedRunSpecsAcceptLabelsOrFolderNamesAndRejectAmbiguity(tmp_path):
    runs = resolveTrackedRuns([str(tmp_path / "playerRun"), f"ballPlayer={tmp_path / 'other'}"])
    assert runs == {"playerRun": (tmp_path / "playerRun").resolve(), "ballPlayer": (tmp_path / "other").resolve()}
    for specs in ([], [str(tmp_path / "a" / "same"), str(tmp_path / "b" / "same")],
                  [f"bad label={tmp_path}"], [f"x={tmp_path}", f"y={tmp_path}"]):
        with pytest.raises(ValueError):
            resolveTrackedRuns(specs)
