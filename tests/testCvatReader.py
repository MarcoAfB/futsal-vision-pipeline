import math
from pathlib import Path

import pytest

from futsal.cvatReader import readCvat


def writeXml(tmp_path: Path, contents: str) -> Path:
    annotationPath = tmp_path / "annotations.xml"
    annotationPath.write_text(contents, encoding="utf-8")
    return annotationPath


def testTaskTracksPreserveExplicitFramesAndAttributes(tmp_path):
    annotationPath = writeXml(
        tmp_path,
        """<annotations><version>1.1</version>
        <meta><task><id>42</id><source>C:\\recordings\\match.mp4</source>
          <size>100</size><original_size><width>640</width><height>360</height></original_size>
          <labels><label><name>Player A</name></label><label><name>Ball</name></label></labels>
        </task></meta>
        <track id="7" label="Player A">
          <box frame="2" xtl="10" ytl="20" xbr="30" ybr="50" occluded="1">
            <attribute name="role">goalkeeper</attribute>
          </box>
          <box frame="3" outside="1"/>
          <box frame="8" xtl="11" ytl="21" xbr="31" ybr="51"/>
        </track>
        <track id="9" label="Ball"><points frame="5" points="30.5,40.5;32,42"/></track>
        </annotations>""",
    )
    originalBytes = annotationPath.read_bytes()

    document = readCvat(annotationPath)

    assert [shape.frameIndex for shape in document.shapes] == [2, 8, 5]
    assert document.outsideShapeCount == 1
    assert document.defaultDimensions == (640, 360)
    assert document.dimensions == {}
    assert document.declaredLabels == ["Player A", "Ball"]
    assert document.metadata == {
        "annotationMode": "tracks",
        "formatVersion": "1.1",
        "taskId": "42",
        "source": "match.mp4",
    }
    assert document.shapes[0].coordinates == (10, 20, 30, 50)
    assert document.shapes[0].trackId == "7"
    assert document.shapes[0].occluded is True
    assert document.shapes[0].attributes == {"role": "goalkeeper"}
    assert document.shapes[2].shapeType == "points"
    assert document.shapes[2].coordinates == (30.5, 40.5, 32, 42)
    assert annotationPath.read_bytes() == originalBytes


def testJobMetadataDoesNotExpandDeclaredFrameRange(tmp_path):
    document = readCvat(
        writeXml(
            tmp_path,
            """<annotations><meta><job><id>8</id><source>match.mp4</source>
              <start_frame>0</start_frame><stop_frame>99</stop_frame>
              <original_size><width>1920</width><height>1080</height></original_size>
              <labels><label><name>Court corner</name></label></labels>
            </job></meta><track id="2" label="Court corner">
              <points frame="10" points="1,2"/><points frame="20" points="3,4"/>
            </track></annotations>""",
        )
    )

    assert [shape.frameIndex for shape in document.shapes] == [10, 20]
    assert document.metadata["jobId"] == "8"
    assert document.declaredLabels == ["Court corner"]
    assert document.defaultDimensions == (1920, 1080)


def testJobExportReadsOriginalSizeBesideJobMetadata(tmp_path):
    document = readCvat(
        writeXml(
            tmp_path,
            """<annotations><meta><job><id>8</id>
              <labels><label><name>Ball</name></label></labels>
            </job><dumped>2026-01-01</dumped>
            <original_size><width>1920</width><height>1080</height></original_size>
            </meta><track id="2" label="Ball">
              <ellipse frame="10" cx="20" cy="30" rx="3" ry="4"/>
            </track></annotations>""",
        )
    )

    assert document.defaultDimensions == (1920, 1080)
    assert document.declaredLabels == ["Ball"]
    assert document.metadata["jobId"] == "8"
    assert document.metadata["annotationMode"] == "tracks"


@pytest.mark.parametrize("nestedWidth", [1920, 1280])
def testRootAndNestedDimensionsMustAgree(tmp_path, nestedWidth):
    annotationPath = writeXml(
        tmp_path,
        f"""<annotations><meta>
        <task><original_size><width>{nestedWidth}</width><height>1080</height></original_size></task>
        <original_size><width>1920</width><height>1080</height></original_size>
        </meta></annotations>""",
    )

    if nestedWidth == 1920:
        assert readCvat(annotationPath).defaultDimensions == (1920, 1080)
    else:
        with pytest.raises(ValueError, match="conflicting default image dimensions"):
            readCvat(annotationPath)


def testImageModeKeepsExportIdsNamesAndIndividualDimensions(tmp_path):
    document = readCvat(
        writeXml(
            tmp_path,
            """<annotations>
            <image id="0" name="Jogo1_frame_1700.jpg" width="640" height="360">
              <box label="Ball" xtl="0" ytl="1" xbr="2" ybr="3"/>
              <points label="Corner" points="20,30" occluded="1"/>
            </image>
            <image id="4" name="/private/images/empty.jpg" width="1920" height="1080"/>
            </annotations>""",
        )
    )

    assert [shape.frameIndex for shape in document.shapes] == [0, 0]
    assert [shape.label for shape in document.shapes] == ["Ball", "Corner"]
    assert all(shape.trackId is None for shape in document.shapes)
    assert document.dimensions == {0: (640, 360), 4: (1920, 1080)}
    assert document.defaultDimensions is None
    assert document.metadata["annotationMode"] == "images"
    assert document.metadata["imageNames"] == {0: "Jogo1_frame_1700.jpg", 4: "empty.jpg"}


def testInvalidBoundsAndNonfiniteCoordinatesRemainAvailableForInventory(tmp_path):
    document = readCvat(
        writeXml(
            tmp_path,
            """<annotations><track id="0" label="Player">
              <box frame="0" xtl="30" ytl="-1" xbr="10" ybr="-1"/>
              <points frame="1" points="NaN,inf;-inf,4"/>
            </track></annotations>""",
        )
    )

    assert document.shapes[0].coordinates == (30, -1, 10, -1)
    values = document.shapes[1].coordinates
    assert math.isnan(values[0])
    assert values[1:] == (math.inf, -math.inf, 4)


def testOutsideShapesAreCountedWithoutInterpretingTheirCoordinates(tmp_path):
    document = readCvat(
        writeXml(
            tmp_path,
            """<annotations><track id="0" label="Ball">
              <box frame="0" outside="1"/><points frame="1" outside="1" points="bad"/>
            </track><image id="2" width="10" height="10">
              <box label="Ball" outside="1"/>
            </image></annotations>""",
        )
    )

    assert document.outsideShapeCount == 3
    assert document.shapes == []
    assert document.metadata["annotationMode"] == "mixed"


def testEllipseCoordinatesAndInvalidRadiusArePreservedWithoutConversion(tmp_path):
    document = readCvat(
        writeXml(
            tmp_path,
            """<annotations><track id="0" label="Ball">
              <ellipse frame="0" cx="12.5" cy="20" rx="3" ry="4"/>
              <ellipse frame="1" cx="NaN" cy="20" rx="-3" ry="0"/>
              <ellipse frame="2" outside="1"/>
            </track></annotations>""",
        )
    )

    assert document.shapes[0].shapeType == "ellipse"
    assert document.shapes[0].coordinates == (12.5, 20, 3, 4)
    assert math.isnan(document.shapes[1].coordinates[0])
    assert document.shapes[1].coordinates[1:] == (20, -3, 0)
    assert document.outsideShapeCount == 1


@pytest.mark.parametrize(
    ("shapeXml", "message"),
    [
        ('<box frame="3" xtl="wrong" ytl="0" xbr="1" ybr="1"/>', "numeric coordinate xtl"),
        ('<box frame="3" xtl="0" ytl="0" xbr="1"/>', "missing or empty ybr"),
        ('<points frame="3" points="1,2,3"/>', "x,y pair"),
        ('<points frame="3" points="1,"/>', "missing or empty y"),
        ('<points frame="3" points="1,2;"/>', "x,y pair"),
        ('<polygon frame="3" points="1,2;3,4;5,6"/>', "unsupported shape type"),
        ('<points frame="3" points="1,2" occluded="yes"/>', "occluded must be 0 or 1"),
    ],
)
def testMalformedActiveShapeRaisesContextualError(tmp_path, shapeXml, message):
    annotationPath = writeXml(
        tmp_path,
        f'<annotations><track id="7" label="Ball">{shapeXml}</track></annotations>',
    )

    with pytest.raises(ValueError, match=message) as error:
        readCvat(annotationPath)

    assert "annotations.xml" in str(error.value)
    assert "track '7'" in str(error.value)
    assert "frame 3" in str(error.value)
    assert str(tmp_path) not in str(error.value)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("<other/>", "expected annotations root"),
        ("<annotations>", "malformed CVAT XML"),
        ('<annotations><image id="-1" width="1" height="1"/></annotations>', "at least 0"),
        ('<annotations><image id="1" width="0" height="1"/></annotations>', "at least 1"),
        (
            '<annotations><track id="1" label="Ball"><points frame="1.5" '
            'points="1,2"/></track></annotations>',
            "invalid integer frame",
        ),
        (
            '<annotations><image id="0" width="1" height="1"/>'
            '<image id="0" width="1" height="1"/></annotations>',
            "duplicate image id",
        ),
        (
            '<annotations><track id="1" label="Ball"><points frame="1" points="1,2">'
            '<attribute name="role">a</attribute><attribute name="role">b</attribute>'
            "</points></track></annotations>",
            "duplicate attribute name",
        ),
    ],
)
def testMalformedStructureDoesNotSilentlyDiscardAnnotations(tmp_path, contents, message):
    with pytest.raises(ValueError, match=message):
        readCvat(writeXml(tmp_path, contents))


def testConflictingTaskAndJobDimensionsAreRejected(tmp_path):
    with pytest.raises(ValueError, match="conflicting default image dimensions"):
        readCvat(
            writeXml(
                tmp_path,
                """<annotations><meta>
                <task><original_size><width>100</width><height>50</height></original_size></task>
                <job><original_size><width>200</width><height>100</height></original_size></job>
                </meta></annotations>""",
            )
        )


def testEmptyDocumentDoesNotInventFrames(tmp_path):
    document = readCvat(writeXml(tmp_path, "<annotations/>"))

    assert document.shapes == []
    assert document.declaredLabels == []
    assert document.dimensions == {}
    assert document.defaultDimensions is None
    assert document.metadata == {"annotationMode": "empty"}
    assert document.outsideShapeCount == 0
