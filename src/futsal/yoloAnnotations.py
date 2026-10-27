import hashlib
import math
from collections import Counter

from futsal.catalog import resolveSourcePath


def coordinateFinding(fields):
    try:
        coordinates = [float(value) for value in fields]
    except ValueError:
        return "invalidCoordinates", "Coordinates must be numeric."
    if not all(math.isfinite(value) for value in coordinates):
        return "nonFiniteCoordinates", "Coordinates must be finite."
    x, y, width, height = coordinates
    if width <= 0 or height <= 0:
        return "degenerateBox", "Bounding box width and height must be positive."
    if not (
        0 <= x - width / 2 <= x + width / 2 <= 1 and 0 <= y - height / 2 <= y + height / 2 <= 1
    ):
        return "coordinatesOutOfBounds", "Normalized box edges fall outside [0, 1]."
    return None


def readYoloAnnotations(relativePath, source, labelEntries, dataRoot, issues):
    result = {
        "path": relativePath,
        "sha256": None,
        "byteSize": None,
        "empty": None,
        "sourceClassCounts": {},
        "canonicalClassCounts": {},
        "observations": [],
    }
    try:
        content = resolveSourcePath(dataRoot, relativePath).read_bytes()
        result.update(sha256=hashlib.sha256(content).hexdigest(), byteSize=len(content))
        text = content.decode("utf-8-sig")
    except (OSError, ValueError, UnicodeError) as error:
        issues.append(
            {
                "severity": "error",
                "code": "annotationUnreadable",
                "sourceId": source["sourceId"],
                "labelPath": relativePath,
                "message": f"Cannot read annotation file ({type(error).__name__}).",
            }
        )
        return result
    result["empty"] = not text.strip()
    sourceCounts, canonicalCounts = Counter(), Counter()
    for lineNumber, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        context = {
            "severity": "error",
            "sourceId": source["sourceId"],
            "labelPath": relativePath,
            "lineNumber": lineNumber,
        }
        if len(fields) != 5:
            issues.append(
                {
                    **context,
                    "code": "annotationFieldCount",
                    "message": f"Expected 5 fields, observed {len(fields)}.",
                }
            )
            continue
        try:
            classId = int(fields[0])
            if classId < 0:
                raise ValueError
        except ValueError:
            issues.append(
                {
                    **context,
                    "code": "invalidClassId",
                    "message": "Source class ID must be a nonnegative integer.",
                }
            )
            continue
        sourceCounts[str(classId)] += 1
        entry = labelEntries.get(classId)
        if entry is None:
            issues.append(
                {
                    **context,
                    "code": "unknownClassId",
                    "sourceClassId": classId,
                    "message": "No declared label mapping exists for this class.",
                }
            )
        elif entry["action"] == "map":
            canonicalCounts[entry["targetClass"]] += 1
        finding = coordinateFinding(fields[1:])
        if finding:
            issues.append({**context, "code": finding[0], "message": finding[1]})
        coordinates = []
        for value in fields[1:]:
            try:
                number = float(value)
                coordinates.append(number if math.isfinite(number) else value)
            except ValueError:
                coordinates.append(value)
        result["observations"].append(
            {
                "lineNumber": lineNumber,
                "sourceClassId": classId,
                "sourceLabel": entry.get("sourceLabel") if entry else None,
                "mappingAction": entry["action"] if entry else "unmapped",
                "canonicalClass": entry.get("targetClass") if entry else None,
                "semanticClass": entry.get("semanticClass") if entry else None,
                "shapeType": "box",
                "coordinates": coordinates,
                "geometryFindings": [finding[0]] if finding else [],
            }
        )
    result["sourceClassCounts"] = dict(sorted(sourceCounts.items()))
    result["canonicalClassCounts"] = dict(sorted(canonicalCounts.items()))
    return result


def readLabels(relativePath, source, labelEntries, dataRoot, issues):
    report = readYoloAnnotations(relativePath, source, labelEntries, dataRoot, issues)
    del report["observations"]
    return report
