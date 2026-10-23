import math

from futsal.cvatReader import CvatShape


def addIssue(issues: list[dict], severity: str, code: str, message: str, **context) -> None:
    issues.append({"severity": severity, "code": code, "message": message, **context})


def geometryFindings(shape: CvatShape, dimensions: tuple[int, int] | None) -> list[str]:
    coordinates = shape.coordinates
    if not all(math.isfinite(value) for value in coordinates):
        return ["nonfiniteGeometry"]
    if shape.shapeType == "box":
        left, top, right, bottom = coordinates
        if right <= left or bottom <= top:
            return ["degenerateGeometry"]
    elif shape.shapeType == "ellipse":
        centerX, centerY, radiusX, radiusY = coordinates
        if radiusX <= 0 or radiusY <= 0:
            return ["degenerateGeometry"]
        left, top, right, bottom = (
            centerX - radiusX,
            centerY - radiusY,
            centerX + radiusX,
            centerY + radiusY,
        )
    else:
        left, right = min(coordinates[::2]), max(coordinates[::2])
        top, bottom = min(coordinates[1::2]), max(coordinates[1::2])
    if dimensions is None:
        return ["dimensionsUnavailable"]
    width, height = dimensions
    if left < 0 or top < 0 or right > width or bottom > height:
        return ["geometryOutOfBounds"]
    return []


def auditShape(
    shape: CvatShape,
    labelEntries: dict[str, dict],
    dimensions: tuple[int, int] | None,
    context: dict,
    issues: list[dict],
) -> dict:
    mapping = labelEntries.get(shape.label.strip().casefold())
    if mapping is None:
        addIssue(
            issues,
            "error",
            "unmappedLabel",
            "No declared label mapping exists.",
            sourceLabel=shape.label,
            **context,
        )
    findings = geometryFindings(shape, dimensions)
    for code in findings:
        severity = "warning" if code == "dimensionsUnavailable" else "error"
        addIssue(
            issues,
            severity,
            code,
            "Source geometry requires review; it was not corrected.",
            sourceLabel=shape.label,
            trackId=shape.trackId,
            **context,
        )
    return {
        "sourceLabel": shape.label,
        "trackId": shape.trackId,
        "shapeType": shape.shapeType,
        "coordinates": [
            value if math.isfinite(value) else str(value) for value in shape.coordinates
        ],
        "occluded": shape.occluded,
        "attributes": shape.attributes,
        "mappingAction": mapping["action"] if mapping else "unmapped",
        "canonicalClass": mapping.get("targetClass") if mapping else None,
        "semanticClass": mapping.get("semanticClass") if mapping else None,
        "geometryFindings": findings,
    }
