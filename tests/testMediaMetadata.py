import math

import cv2
import numpy as np
import pytest

from futsal.mediaMetadata import readImageMetadata, readVideoMetadata


class FakeCapture:
    def __init__(self, properties=None, *, opened=True, readError=None, releaseError=None):
        self.properties = properties or {}
        self.opened = opened
        self.readError = readError
        self.releaseError = releaseError
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, propertyId):
        if self.readError is not None:
            raise self.readError
        return self.properties.get(propertyId, 0)

    def release(self):
        self.released = True
        if self.releaseError is not None:
            raise self.releaseError

    def read(self):
        raise AssertionError("Metadata inspection must not decode video frames.")


def installCapture(tmp_path, monkeypatch, capture):
    videoPath = tmp_path / "match.mp4"
    videoPath.write_bytes(b"synthetic container placeholder")
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    return videoPath


def testImageMetadataUsesRealEncodedDimensionsWithoutRewriting(tmp_path):
    imagePath = tmp_path / "sample.png"
    success, encoded = cv2.imencode(".png", np.zeros((17, 31, 4), dtype=np.uint8))
    assert success
    originalBytes = encoded.tobytes()
    imagePath.write_bytes(originalBytes)

    assert readImageMetadata(imagePath) == {
        "byteSize": len(originalBytes),
        "width": 31,
        "height": 17,
    }
    assert imagePath.read_bytes() == originalBytes


def testVideoMetadataOnlyProbesPropertiesAndReleasesCapture(tmp_path, monkeypatch):
    capture = FakeCapture(
        {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 25,
            cv2.CAP_PROP_FRAME_COUNT: 250,
        }
    )
    videoPath = installCapture(tmp_path, monkeypatch, capture)
    originalBytes = videoPath.read_bytes()

    assert readVideoMetadata(videoPath) == {
        "byteSize": len(originalBytes),
        "width": 1920,
        "height": 1080,
        "fps": 25.0,
        "frameCount": 250,
        "durationSeconds": 10.0,
        "metadataProvider": "OpenCV",
    }
    assert capture.released
    assert videoPath.read_bytes() == originalBytes


@pytest.mark.parametrize("fps", [0, -1, math.nan, math.inf, -math.inf, None])
def testMissingOrInvalidFpsRemainsUnknown(tmp_path, monkeypatch, fps):
    capture = FakeCapture({cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_FRAME_COUNT: 100})

    metadata = readVideoMetadata(installCapture(tmp_path, monkeypatch, capture))

    assert metadata["fps"] is None
    assert metadata["durationSeconds"] is None
    assert metadata["frameCount"] == 100
    assert metadata["width"] is None
    assert metadata["height"] is None
    assert capture.released


@pytest.mark.parametrize("frameCount", [0, -3, math.nan, math.inf, 3.5])
def testInvalidFrameCountDoesNotProduceDuration(tmp_path, monkeypatch, frameCount):
    capture = FakeCapture({cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: frameCount})

    metadata = readVideoMetadata(installCapture(tmp_path, monkeypatch, capture))

    assert metadata["frameCount"] is None
    assert metadata["durationSeconds"] is None
    assert capture.released


@pytest.mark.parametrize("opened", [True, False])
def testCaptureIsReleasedAfterReadOrOpenFailure(tmp_path, monkeypatch, opened):
    capture = FakeCapture(opened=opened, readError=cv2.error("property read failed"))
    videoPath = installCapture(tmp_path, monkeypatch, capture)

    with pytest.raises(OSError) as error:
        readVideoMetadata(videoPath)

    assert capture.released
    assert videoPath.name in str(error.value)
    assert str(tmp_path) not in str(error.value)


def testInvalidImageRaisesContextualError(tmp_path):
    imagePath = tmp_path / "broken.png"
    imagePath.write_bytes(b"not an image")

    with pytest.raises(OSError, match="broken.png.*valid image") as error:
        readImageMetadata(imagePath)

    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize("reader", [readImageMetadata, readVideoMetadata])
def testMissingFileRaisesContextualError(tmp_path, reader):
    with pytest.raises(OSError, match="missing.mp4.*file metadata") as error:
        reader(tmp_path / "missing.mp4")

    assert str(tmp_path) not in str(error.value)
