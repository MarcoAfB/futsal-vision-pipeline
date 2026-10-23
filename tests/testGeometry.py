import pytest

from futsal.annotationAudit import geometryFindings
from futsal.cvatReader import CvatShape


@pytest.mark.parametrize(
    ("shapeType", "coordinates", "expected"),
    [
        ("box", (1, 1, 1, 3), "degenerateGeometry"),
        ("box", (-1, 1, 2, 3), "geometryOutOfBounds"),
        ("ellipse", (5, 5, -1, 2), "degenerateGeometry"),
        ("ellipse", (0, 0, 2, 2), "geometryOutOfBounds"),
        ("points", (101, 5), "geometryOutOfBounds"),
        ("points", (float("inf"), 5), "nonfiniteGeometry"),
    ],
)
def testInvalidGeometryIsNotClamped(shapeType, coordinates, expected):
    shape = CvatShape(0, "Bola", shapeType, coordinates, "1", False, {})
    assert geometryFindings(shape, (100, 100)) == [expected]
    assert shape.coordinates == coordinates
