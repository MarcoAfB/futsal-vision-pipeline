from pathlib import Path

import cv2


def readSelectedFrames(videoPath, frameIndices):
    requested = set(frameIndices)
    if not requested:
        return
    capture = cv2.VideoCapture(str(videoPath))
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {Path(videoPath).name}.")
        for frameIndex in range(max(requested) + 1):
            if not capture.grab():
                raise OSError(f"Video ended before requested frame {frameIndex}: {Path(videoPath).name}.")
            if frameIndex in requested:
                success, image = capture.retrieve()
                if not success or image is None:
                    raise OSError(f"Cannot decode frame {frameIndex}: {Path(videoPath).name}.")
                yield frameIndex, image
    except cv2.error as error:
        raise OSError(f"Video decoding failed: {Path(videoPath).name}.") from error
    finally:
        capture.release()
