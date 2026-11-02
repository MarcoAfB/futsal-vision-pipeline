import math
import stat
from pathlib import Path

import cv2


def fileByteSize(mediaPath: Path) -> int:

    try:
        fileStat = mediaPath.stat()
    except OSError as error:
        raise OSError(
            f"{mediaPath.name}: cannot read file metadata ({type(error).__name__})."
        ) from error
    if not stat.S_ISREG(fileStat.st_mode):
        raise ValueError(f"{mediaPath.name}: expected a regular media file.")
    return fileStat.st_size


def positiveNumber(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def positiveInteger(value: object) -> int | None:
    number = positiveNumber(value)
    return int(number) if number is not None and number.is_integer() else None


def readVideoMetadata(videoPath: Path) -> dict[str, object]:

    videoPath = Path(videoPath)
    byteSize = fileByteSize(videoPath)
    capture = cv2.VideoCapture(str(videoPath))
    try:
        if not capture.isOpened():
            raise OSError(f"{videoPath.name}: OpenCV could not open the video metadata reader.")
        width = positiveInteger(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = positiveInteger(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = positiveNumber(capture.get(cv2.CAP_PROP_FPS))
        frameCount = positiveInteger(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        durationSeconds = (
            positiveNumber(frameCount / fps) if frameCount is not None and fps is not None else None
        )
        result = {
            "byteSize": byteSize,
            "width": width,
            "height": height,
            "fps": fps,
            "frameCount": frameCount,
            "durationSeconds": durationSeconds,
            "metadataProvider": "OpenCV",
        }
        return result
    except cv2.error as error:
        raise OSError(f"{videoPath.name}: OpenCV failed while reading video metadata.") from error
    finally:
        capture.release()


def readImageMetadata(imagePath: Path) -> dict[str, int]:

    imagePath = Path(imagePath)
    byteSize = fileByteSize(imagePath)
    try:
        image = cv2.imread(str(imagePath), cv2.IMREAD_UNCHANGED)
    except cv2.error as error:
        raise OSError(f"{imagePath.name}: OpenCV failed while reading the image.") from error
    if image is None or image.ndim < 2 or min(image.shape[:2]) <= 0:
        raise OSError(f"{imagePath.name}: OpenCV could not decode a valid image.")
    height, width = image.shape[:2]
    return {"byteSize": byteSize, "width": int(width), "height": int(height)}
