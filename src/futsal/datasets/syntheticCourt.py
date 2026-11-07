import math
from pathlib import Path

from futsal.datasets.courtRendering import LabeledReference, SynthesisModel
from futsal.inventory.catalog import readYaml

projectRoot = Path(__file__).resolve().parents[3]


defaultConfig = projectRoot / "configs/syntheticCourt.yaml"


pointBoxes = {"widthFraction": 1 / 64, "heightFraction": 1 / 36}


def courtClassNames():
    labels, _ = readYaml(projectRoot / "configs/labelMap.yaml")
    return [item["name"] for item in labels["canonicalClasses"]][5:]


def synthesisModel(config):
    geometry = config["geometry"]
    names = courtClassNames()
    if (
        len(names) != 20
        or set(geometry)
        != {
            "length",
            "width",
            "goalWidth",
            "penaltyDistance",
            "secondPenaltyDistance",
            "centerCircleRadius",
            "references",
        }
        or set(geometry["references"]) != set(names)
    ):
        raise ValueError(
            "Synthesis geometry must define exactly the twenty canonical court references."
        )
    if any(
        type(value) not in {float, int} or not math.isfinite(value) or value <= 0
        for key, value in geometry.items()
        if key != "references"
    ):
        raise ValueError("Court dimensions must be finite and positive.")
    length, width = geometry["length"], geometry["width"]
    if (
        geometry["goalWidth"] >= width
        or geometry["penaltyDistance"] >= geometry["secondPenaltyDistance"]
        or geometry["secondPenaltyDistance"] >= length / 2
        or geometry["centerCircleRadius"] >= width / 2
    ):
        raise ValueError("Court geometry has incompatible goal, penalty or circle dimensions.")
    references = []
    for index, name in enumerate(names):
        coordinates = geometry["references"][name]
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != 2
            or any(
                type(v) not in {int, float} or not math.isfinite(v) or not 0 <= v <= bound
                for v, bound in zip(coordinates, (length, width), strict=True)
            )
        ):
            raise ValueError(f"Invalid metric court reference: {name}.")
        references.append(LabeledReference(index, name, name, tuple(coordinates)))
    if len({r.coordinate for r in references}) != 20:
        raise ValueError("Court references must have distinct coordinates.")
    centerX, centerY = length / 2, width / 2
    upperPost, lowerPost = centerY - geometry["goalWidth"] / 2, centerY + geometry["goalWidth"] / 2
    penalty, second, radius = (
        geometry["penaltyDistance"],
        geometry["secondPenaltyDistance"],
        geometry["centerCircleRadius"],
    )
    expected = [
        (0, 0),
        (0, width),
        (0, upperPost),
        (0, lowerPost),
        (penalty, centerY),
        (second, centerY),
        (centerX, 0),
        (centerX, width),
        (centerX, centerY - radius),
        (centerX, centerY + radius),
        (centerX - 10, width),
        (centerX - 5, width),
        (centerX + 5, width),
        (centerX + 10, width),
        (length, 0),
        (length, width),
        (length, upperPost),
        (length, lowerPost),
        (length - penalty, centerY),
        (length - second, centerY),
    ]
    if [r.coordinate for r in references] != expected:
        raise ValueError(
            "Metric references must coincide with the rendered court markings in canonical order."
        )
    return SynthesisModel(
        "futsalCourt",
        **{k: v for k, v in geometry.items() if k != "references"},
        references=tuple(references),
        classNames=dict(enumerate(names)),
    )
