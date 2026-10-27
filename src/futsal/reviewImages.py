import math

import cv2


def renderReviewImage(image, sample):
    preview = image.copy()
    height, width = preview.shape[:2]
    for index, annotation in enumerate(sample["annotations"]):
        values = annotation["coordinates"]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            continue
        shape = annotation["shapeType"]
        if sample["coordinateSystem"] == "normalizedCenterBoxes":
            x, y, boxWidth, boxHeight = values
            values = [(x - boxWidth / 2) * width, (y - boxHeight / 2) * height,
                      (x + boxWidth / 2) * width, (y + boxHeight / 2) * height]
        limit = 3 * max(width, height)
        pixels = [int(max(-limit, min(limit, value))) for value in values]
        if shape == "box":
            left, top, right, bottom = pixels
            cv2.rectangle(preview, (left, top), (right, bottom), (0, 180, 255), 2)
        elif shape == "ellipse":
            x, y, radiusX, radiusY = pixels
            cv2.ellipse(preview, (x, y), (max(1, radiusX), max(1, radiusY)),
                        0, 0, 360, (0, 180, 255), 2)
        elif shape == "points":
            for x, y in zip(pixels[::2], pixels[1::2], strict=True):
                cv2.circle(preview, (x, y), 5, (0, 180, 255), 2)
        name = annotation.get("canonicalClass") or annotation.get("semanticClass") or "unmapped"
        cv2.putText(preview, f"{index}: {name}",
                    (max(0, min(width - 1, pixels[0])), max(15, min(height - 1, pixels[1]))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(preview, sample["recordId"], (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return preview
