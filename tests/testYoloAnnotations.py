import hashlib
import json

from futsal.yoloAnnotations import readLabels


def testSourceClassesAndInvalidObservationsKeepTheirProvenance(tmp_path):
    content = (
        "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.1 0.1\n"
        "0 nan 0.5 0.2 0.2\n0 0.5 0.5 0 0.2\n99 0.5 0.5 0.2 0.2\n"
        "0 0.1 0.5 0.3 0.2\n0 0.5 0.5\n0 text 0.5 0.2 0.2\n"
    )
    path = tmp_path / "labels.txt"
    path.write_text(content, encoding="utf-8")
    entries = {
        0: {"action": "map", "targetClass": "ball"},
        1: {"action": "preserve", "targetClass": None},
    }
    issues = []
    report = readLabels("labels.txt", {"sourceId": "example"}, entries, tmp_path, issues)
    assert report["sourceClassCounts"] == {"0": 5, "1": 1, "99": 1}
    assert report["canonicalClassCounts"] == {"ball": 5}
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert {item["code"] for item in issues} == {
        "nonFiniteCoordinates",
        "degenerateBox",
        "unknownClassId",
        "coordinatesOutOfBounds",
        "annotationFieldCount",
        "invalidCoordinates",
    }
    assert {item["lineNumber"] for item in issues} == set(range(3, 9))
    assert path.read_text(encoding="utf-8") == content
    json.dumps(report, allow_nan=False)
