from pathlib import Path

import pytest
import yaml

from futsal.geometry.courtModel import landmarkNames, loadCourtGeometry, matchCourt


def writeGeometry(tmp_path, **changes):
    config = yaml.safe_load(Path("configs/courtGeometry.yaml").read_text())
    config.update(changes)
    path = tmp_path / "courtGeometry.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def testDefaultCourtMatchesCanonicalOrderAndSyntheticReferences():
    labels = yaml.safe_load(Path("configs/labelMap.yaml").read_text())["canonicalClasses"]
    synthetic = yaml.safe_load(Path("configs/syntheticCourt.yaml").read_text())["geometry"]["references"]
    court = matchCourt(loadCourtGeometry("configs/courtGeometry.yaml"), "match01")

    assert list(landmarkNames) == [item["name"] for item in labels if item["geometry"] == "point"]
    assert court["landmarks"] == {name: tuple(synthetic[name]) for name in landmarkNames}
    assert (court["length"], court["width"], court["dimensionSource"]) == (40.0, 20.0, "nominalDefault")


def testSuppliedDimensionsMoveCourtLinesButKeepRegulationMarkings(tmp_path):
    path = writeGeometry(tmp_path, matchDimensions={"match11": {"length": 38.0, "width": 19.0}})
    geometry = loadCourtGeometry(path, 36.0, 18.0)
    measured, supplied = matchCourt(geometry, "match11"), matchCourt(geometry, "match01")

    assert measured["dimensionSource"] == "matchConfig"
    assert supplied["dimensionSource"] == "commandLine"
    assert measured["landmarks"]["lowerCornerSideB"] == (38.0, 19.0)
    assert measured["landmarks"]["penaltyMarkSideB"] == (32.0, 9.5)
    assert measured["landmarks"]["upperHalfwayLinePoint"] == (19.0, 6.5)
    assert measured["landmarks"]["substitutionPoint1"] == (9.0, 19.0)
    assert supplied["landmarks"]["secondPenaltyMarkSideA"] == (10.0, 9.0)


@pytest.mark.parametrize("length,width", [(45.0, 20.0), (40.0, 12.0), (40.0, None)])
def testImplausibleOrPartialDimensionsAreRejected(length, width):
    with pytest.raises(ValueError):
        loadCourtGeometry("configs/courtGeometry.yaml", length, width)
