import cv2
import numpy as np

from futsal.datasets.mediaStore import readBindings, resolveAsset
from futsal.evaluation.homographyMetrics import acceptedStatuses, frameP90
from futsal.geometry.courtModel import courtPolylines, matchCourt
from futsal.geometry.courtProjection import projectPoints
from futsal.inventory.catalog import hashFile

white, green, red, yellow, magenta = (255, 255, 255), (80, 220, 80), (60, 60, 230), (0, 220, 255), (255, 0, 255)


def selectExamples(rows):
    chosen = {}
    for matchId in sorted({row["matchId"] for row in rows}):
        verified = [row for row in rows if row["matchId"] == matchId and row["reference"]["status"] == "verified"]
        accepted = sorted((row for row in verified if row["prediction"]["status"] in acceptedStatuses
                           and frameP90(row) is not None), key=lambda row: (frameP90(row), row["recordId"]))
        rejected = [row for row in verified if row["prediction"]["status"] == "rejected"]
        for kind, row in (("medianAccepted", accepted[len(accepted) // 2] if accepted else None),
                          ("worstAccepted", accepted[-1] if accepted else None),
                          ("rejected", rejected[0] if rejected else None)):
            if row is not None and row["recordId"] not in chosen:
                chosen[row["recordId"]] = (kind, row)
    return list(chosen.values())


def densify(polyline, spacing=0.25):
    points = []
    for start, end in zip(polyline[:-1], polyline[1:], strict=True):
        count = max(2, int(np.ceil(np.linalg.norm(end - start) / spacing)) + 1)
        points.extend(np.linspace(start, end, count)[:-1])
    return np.array([*points, polyline[-1]])


def drawCourt(image, matrix, court, markings, color):
    limit = 4 * max(image.shape[:2])
    for polyline in courtPolylines(court, markings):
        projected, scale = projectPoints(matrix, densify(polyline))
        valid = (scale > 0) & np.isfinite(projected).all(1) & (np.abs(projected) < limit).all(1)
        segments, current = [], []
        for point, usable in zip(projected, valid, strict=True):
            if usable:
                current.append(point)
            elif current:
                segments.append(current)
                current = []
        segments.append(current)
        for segment in segments:
            if len(segment) >= 2:
                cv2.polylines(image, [np.round(segment).astype(np.int32)], False, color, 2, cv2.LINE_AA)


def drawTopView(image, court, markings, athleteRows):
    height, width = image.shape[:2]
    panelWidth = width // 4
    scale = (panelWidth - 20) / court["length"]
    panelHeight = int(court["width"] * scale) + 20
    left, top = width - panelWidth - 10, height - panelHeight - 10
    image[top:top + panelHeight, left:left + panelWidth] //= 3

    def toPanel(point):
        return (int(round(left + 10 + point[0] * scale)), int(round(top + 10 + point[1] * scale)))

    def place(point):
        x, y = toPanel(point)
        return (x, y) if left <= x < left + panelWidth and top <= y < top + panelHeight else None

    for polyline in courtPolylines(court, markings):
        cv2.polylines(image, [np.array([toPanel(point) for point in polyline], dtype=np.int32)], False,
                      (180, 180, 180), 1, cv2.LINE_AA)
    for row in athleteRows:
        truth = None if row["truthMeters"] is None else place(row["truthMeters"])
        if truth is None:
            continue
        predicted = None if row["endToEndPositionMeters"] is None else place(row["endToEndPositionMeters"])
        if predicted is not None:
            cv2.line(image, truth, predicted, yellow, 1, cv2.LINE_AA)
            cv2.circle(image, predicted, 4, yellow, -1, cv2.LINE_AA)
        elif row["endToEndPositionMeters"] is not None:
            cv2.circle(image, truth, 7, red, 2, cv2.LINE_AA)
        cv2.circle(image, truth, 4, white, -1, cv2.LINE_AA)


def drawPoints(image, row):
    reference = row["reference"]
    for point in reference["imagePoints"]:
        x, y = map(int, np.round(point))
        cv2.drawMarker(image, (x, y), magenta, cv2.MARKER_CROSS, 18, 2)
    for item in row["prediction"]["candidates"]:
        center = tuple(int(round(value)) for value in item["image"])
        cv2.circle(image, center, 9, green if item["inlier"] else red, 2, cv2.LINE_AA)
    lines = [f"{row['recordId']}  reference={reference['status']}  prediction={row['prediction']['status']}",
             f"inliers={row['prediction']['inlierCount']}/{row['prediction']['pointCount']}  "
             f"p90={frameP90(row) if frameP90(row) is None else round(frameP90(row), 2)} m  "
             f"reasons={','.join(row['prediction']['reasons']) or 'none'}"]
    for index, text in enumerate(lines):
        position = (20, 40 + index * 36)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, white, 2, cv2.LINE_AA)


def loadImage(mediaStoreDir, imageSha256, bindings):
    path = resolveAsset(mediaStoreDir, imageSha256, bindings=bindings)
    if hashFile(path) != imageSha256:
        raise ValueError(f"Media asset content changed: {imageSha256}.")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Media asset could not be decoded: {imageSha256}.")
    return image


def renderExample(row, athleteRows, geometry, image):
    court = matchCourt(geometry, row["matchId"])
    if row["reference"]["courtToImage"] is not None:
        drawCourt(image, np.asarray(row["reference"]["courtToImage"]), court, geometry["markings"], magenta)
    if row["prediction"]["courtToImage"] is not None:
        color = green if row["prediction"]["status"] in acceptedStatuses else red
        drawCourt(image, np.asarray(row["prediction"]["courtToImage"]), court, geometry["markings"], color)
    drawPoints(image, row)
    drawTopView(image, court, geometry["markings"], athleteRows)
    scale = 1280 / image.shape[1]
    return cv2.resize(image, (1280, int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def writeExamples(rows, athleteRows, geometry, mediaStoreDir, directory):
    """Overlay reference (magenta) and predicted (green accepted, red rejected) court models."""
    directory.mkdir(parents=True)
    bindings = readBindings(mediaStoreDir)
    examples = []
    for kind, row in selectExamples(rows):
        image = loadImage(mediaStoreDir, row["imageSha256"], bindings)
        frameAthletes = [item for item in athleteRows if item["recordId"] == row["recordId"]]
        output = renderExample(row, frameAthletes, geometry, image)
        name = f"{row['matchId']}-{kind}-{row['recordId'].replace(':', '_')}.jpg"
        success, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not success:
            raise ValueError(f"Example image could not be encoded: {name}.")
        (directory / name).write_bytes(encoded.tobytes())
        examples.append({"kind": kind, "recordId": row["recordId"], "matchId": row["matchId"],
                         "path": f"evidence/examples/{name}"})
    return examples
