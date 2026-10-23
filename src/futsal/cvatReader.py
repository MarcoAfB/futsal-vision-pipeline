from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from xml.etree import ElementTree


@dataclass(slots=True)
class CvatShape:
    frameIndex: int
    label: str
    shapeType: str
    coordinates: tuple[float, ...]
    trackId: str | None
    occluded: bool
    attributes: dict[str, str]


@dataclass(slots=True)
class CvatDocument:
    shapes: list[CvatShape]
    declaredLabels: list[str]
    dimensions: dict[int, tuple[int, int]]
    defaultDimensions: tuple[int, int] | None
    metadata: dict[str, object]
    outsideShapeCount: int


def requiredText(value: str | None, fieldName: str, context: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{context}: missing or empty {fieldName}.")
    return value


def integer(value: str | None, fieldName: str, context: str, minimum: int = 0) -> int:
    text = requiredText(value, fieldName, context)
    try:
        number = int(text)
    except ValueError as error:
        raise ValueError(f"{context}: invalid integer {fieldName}={text!r}.") from error
    if number < minimum:
        raise ValueError(f"{context}: {fieldName} must be at least {minimum}.")
    return number


def boolean(element: ElementTree.Element, fieldName: str, context: str) -> bool:
    value = element.get(fieldName, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{context}: {fieldName} must be 0 or 1, received {value!r}.")
    return value == "1"


def coordinate(value: str | None, fieldName: str, context: str) -> float:
    text = requiredText(value, fieldName, context)
    try:
        return float(text)
    except ValueError as error:
        raise ValueError(f"{context}: invalid numeric coordinate {fieldName}={text!r}.") from error


def sourceName(value: str) -> str:

    if (
        PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
        or value.casefold().startswith("file:")
    ):
        return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return value


def readAttributes(element: ElementTree.Element, context: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for attribute in element.findall("attribute"):
        name = requiredText(attribute.get("name"), "attribute name", context)
        if name in attributes:
            raise ValueError(f"{context}: duplicate attribute name {name!r}.")
        attributes[name] = attribute.text or ""
    return attributes


def readShape(
    element: ElementTree.Element,
    frameIndex: int,
    label: str,
    trackId: str | None,
    context: str,
) -> CvatShape | None:
    shapeContext = f"{context}, frame {frameIndex}, {element.tag}"
    if boolean(element, "outside", shapeContext):
        return None
    if element.tag == "box":
        coordinates = tuple(
            coordinate(element.get(name), name, shapeContext)
            for name in ("xtl", "ytl", "xbr", "ybr")
        )
    elif element.tag == "ellipse":
        coordinates = tuple(
            coordinate(element.get(name), name, shapeContext) for name in ("cx", "cy", "rx", "ry")
        )
    elif element.tag == "points":
        pointsText = requiredText(element.get("points"), "points", shapeContext)
        values: list[float] = []
        for pair in pointsText.split(";"):
            parts = pair.split(",")
            if len(parts) != 2:
                raise ValueError(f"{shapeContext}: each point must contain an x,y pair.")
            values.extend(
                coordinate(value, axis, shapeContext)
                for value, axis in zip(parts, ("x", "y"), strict=True)
            )
        coordinates = tuple(values)
    else:
        raise ValueError(f"{shapeContext}: unsupported shape type {element.tag!r}.")
    return CvatShape(
        frameIndex=frameIndex,
        label=label,
        shapeType=element.tag,
        coordinates=coordinates,
        trackId=trackId,
        occluded=boolean(element, "occluded", shapeContext),
        attributes=readAttributes(element, shapeContext),
    )


def readMetadata(root: ElementTree.Element, context: str) -> CvatDocument:
    document = CvatDocument([], [], {}, None, {}, 0)
    hasTracks = root.find("track") is not None
    hasImages = root.find("image") is not None
    document.metadata["annotationMode"] = (
        "mixed"
        if hasTracks and hasImages
        else "tracks"
        if hasTracks
        else "images"
        if hasImages
        else "empty"
    )
    version = root.findtext("version")
    if version is not None:
        document.metadata["formatVersion"] = version.strip()
    for metadataType in ("task", "job"):
        element = root.find(f"meta/{metadataType}")
        if element is None:
            continue
        metadataContext = f"{context}, meta/{metadataType}"
        identifier = element.findtext("id")
        if identifier is not None:
            document.metadata[f"{metadataType}Id"] = identifier.strip()
        source = element.findtext("source")
        if source is not None:
            document.metadata["source"] = sourceName(source.strip())
        for label in element.findall("labels/label"):
            name = requiredText(label.findtext("name"), "label name", metadataContext)
            if name not in document.declaredLabels:
                document.declaredLabels.append(name)
    for sizePath in ("meta/original_size", "meta/task/original_size", "meta/job/original_size"):
        originalSize = root.find(sizePath)
        if originalSize is not None:
            metadataContext = f"{context}, {sizePath}"
            dimensions = (
                integer(originalSize.findtext("width"), "width", metadataContext, 1),
                integer(originalSize.findtext("height"), "height", metadataContext, 1),
            )
            if document.defaultDimensions not in {None, dimensions}:
                raise ValueError(f"{metadataContext}: conflicting default image dimensions.")
            document.defaultDimensions = dimensions
    return document


def readCvat(annotationPath: Path) -> CvatDocument:

    annotationPath = Path(annotationPath)
    context = annotationPath.name
    try:
        root = ElementTree.parse(annotationPath).getroot()
    except ElementTree.ParseError as error:
        raise ValueError(f"{context}: malformed CVAT XML: {error}.") from error
    if root.tag != "annotations":
        raise ValueError(f"{context}: expected annotations root, received {root.tag!r}.")
    document = readMetadata(root, context)
    imageNames: dict[int, str] = {}
    for element in root:
        if element.tag == "track":
            trackId = requiredText(element.get("id"), "track id", context)
            trackContext = f"{context}, track {trackId!r}"
            label = requiredText(element.get("label"), "track label", trackContext)
            for shapeElement in element:
                if shapeElement.tag == "attribute":
                    continue
                frameIndex = integer(shapeElement.get("frame"), "frame", trackContext)
                shape = readShape(shapeElement, frameIndex, label, trackId, trackContext)
                if shape is None:
                    document.outsideShapeCount += 1
                else:
                    document.shapes.append(shape)
        elif element.tag == "image":
            frameIndex = integer(element.get("id"), "image id", context)
            imageContext = f"{context}, image {frameIndex}"
            if frameIndex in document.dimensions:
                raise ValueError(f"{imageContext}: duplicate image id.")
            document.dimensions[frameIndex] = (
                integer(element.get("width"), "width", imageContext, 1),
                integer(element.get("height"), "height", imageContext, 1),
            )
            imageName = element.get("name")
            if imageName is not None:
                imageNames[frameIndex] = sourceName(imageName)
            for shapeElement in element:
                label = requiredText(shapeElement.get("label"), "shape label", imageContext)
                shape = readShape(shapeElement, frameIndex, label, None, imageContext)
                if shape is None:
                    document.outsideShapeCount += 1
                else:
                    document.shapes.append(shape)
    if imageNames:
        document.metadata["imageNames"] = imageNames
    return document
