import csv
from pathlib import Path

import numpy as np


def plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "futsalResearchReportV1", "figure.dpi": 120})
    return plt


def saveFigure(figure, directory, name):
    for extension in ("svg", "png"):
        figure.savefig(directory / f"{name}.{extension}", bbox_inches="tight", dpi=180,
                       metadata={"Creator": "Futsal research reports", "Date": None})
    plotting().close(figure)
    return f"{name}.png"


def classFigure(detection, directory):
    plt = plotting()
    rows = detection["perClass"]
    figure, axes = plt.subplots(1, 2, figsize=(12, max(3, len(rows) * 0.29)), sharey=True,
                                gridspec_kw={"width_ratios": [2, 1]}, layout="constrained")
    y = np.arange(len(rows))
    axes[0].barh(y, [row["ap50To95"] or 0 for row in rows], color="#0072B2")
    axes[0].set_yticks(y, [row["className"] for row in rows])
    axes[0].invert_yaxis()
    maximum = max((row["ap50To95"] or 0 for row in rows), default=0)
    bound = maximum * 1.15 if 0 < maximum < 0.1 else 1
    axes[0].set_xlim(0, bound)
    axes[0].set_xlabel("AP50-95 (zoomed scale)" if bound < 1 else "AP50-95 (supported classes)")
    axes[1].barh(y, [row["support"] for row in rows], color="#009E73")
    axes[1].set_xlabel("Annotated instances")
    for index, row in enumerate(rows):
        if not row["support"]:
            axes[0].text(0.02 * bound, index, "N/A: no reference", va="center", color="#666666")
    figure.suptitle("Detection and reference support by class")
    return saveFigure(figure, directory, "classDetection")


def matchFigure(detection, directory):
    plt = plotting()
    items = list(detection["byMatch"].items())
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    axes[0].bar([key for key, _ in items], [row["groups"]["all"]["ap50To95"] or 0 for _, row in items],
                color="#0072B2")
    axes[0].set_ylabel("Macro AP50-95")
    maximum = max((row["groups"]["all"]["ap50To95"] or 0 for _, row in items), default=0)
    axes[0].set_ylim(0, maximum * 1.15 if 0 < maximum < 0.1 else 1)
    if 0 < maximum < 0.1:
        axes[0].set_ylabel("Macro AP50-95 (zoomed scale)")
    axes[1].bar([key for key, _ in items], [row["imageCount"] for _, row in items], color="#009E73")
    axes[1].set_ylabel("Evaluated images")
    figure.suptitle("Descriptive results by original match")
    return saveFigure(figure, directory, "matchDetection")


def precisionRecallFigure(detection, pointClasses, directory):
    curves = {name: curve for name, curve in detection["prCurves"].items() if name not in pointClasses}
    if not curves or not any(max(curve["precision"], default=0) > 0 for curve in curves.values()):
        return None
    plt = plotting()
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    for name, curve in curves.items():
        axis.plot(curve["recall"], curve["precision"], label=name)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision",
             title="Object precision-recall at IoU 0.50")
    axis.legend(loc="upper right")
    return saveFigure(figure, directory, "objectPrecisionRecall")


def confusionFigure(detection, directory):
    matrix = np.array(detection["confusionMatrix"])
    if matrix[:-1, :-1].sum() + matrix[-1, :-1].sum() == 0:
        return None
    plt = plotting()
    labels = detection["confusionLabels"]
    size = max(5, len(labels) * 0.43)
    figure, axis = plt.subplots(figsize=(size + 2, size), layout="constrained")
    image = axis.imshow(np.log1p(matrix), cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=65, ha="right", fontsize=9)
    axis.set_yticks(range(len(labels)), labels, fontsize=9)
    for row, column in zip(*matrix.nonzero(), strict=True):
        axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
    axis.set(xlabel="Predicted class", ylabel="True class", title="Operational confusion counts (log colour scale)")
    figure.colorbar(image, ax=axis, label="log(1 + count)", shrink=0.7)
    return saveFigure(figure, directory, "confusionCounts")


def pointFigure(points, config, directory):
    if not points["perClass"] or not points["support"]:
        return None
    plt = plotting()
    rows = points["perClass"]
    figure, axes = plt.subplots(1, 2, figsize=(13, max(4, len(rows) * 0.3)),
                                gridspec_kw={"width_ratios": [1, 2]}, layout="constrained")
    tolerances = config["pointTolerances"]
    axes[0].plot([100 * t for t in tolerances], [points["successRates"][str(t)] for t in tolerances],
                 marker="o", color="#0072B2")
    axes[0].axvline(100 * config["primaryPointTolerance"], color="#D55E00", linestyle="--", label="Primary tolerance")
    axes[0].set(xlabel="Tolerance (% image diagonal)", ylabel="Correct / all annotated points",
                ylim=(0, 1), title="Localization success, including missing points")
    axes[0].legend(fontsize=8)
    missing = sum(row["missingCount"] for row in rows)
    axes[0].text(0.5, 0.65, f"{points['support']} annotated points\n{missing} without a prediction",
                 ha="center", transform=axes[0].transAxes, fontsize=10)
    key = str(config["primaryPointTolerance"])
    axes[1].barh(range(len(rows)), [row["successRates"][key] or 0 for row in rows], color="#009E73")
    axes[1].set_yticks(range(len(rows)), [row["className"] for row in rows])
    axes[1].invert_yaxis()
    axes[1].set(xlim=(0, 1), xlabel="Correct / all annotated points", title=f"Per point at {100 * float(key):g}% diagonal")
    for index, row in enumerate(rows):
        if not row["support"]:
            axes[1].text(0.02, index, "N/A", va="center", color="#666666")
        elif row["successRates"][key] == 0:
            axes[1].text(0.02, index, f"0 / {row['support']} correct; {row['missingCount']} missing",
                         va="center", fontsize=8, color="#666666")
    return saveFigure(figure, directory, "pointLocalization")


def historyFigure(path, directory):
    if not Path(path).is_file():
        return None
    with Path(path).open(encoding="utf-8-sig") as stream:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
    if len(rows) < 2:
        return None
    plt = plotting()
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    epochs = [int(row["epoch"]) for row in rows]
    for name in rows[0]:
        if name.startswith(("train/", "val/")) and name.endswith("loss"):
            axes[0].plot(epochs, [float(row[name]) for row in rows], label=name)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Stage-specific losses")
    axes[0].legend(fontsize=7)
    key = "metrics/mAP50-95(B)"
    if key in rows[0]:
        axes[1].plot(epochs, [float(row[key]) for row in rows], color="#0072B2")
    axes[1].set(xlabel="Epoch", ylabel="AP50-95", ylim=(0, 1), title="Training validation history")
    return saveFigure(figure, directory, "trainingHistory")


def stageFigures(detection, points, config, pointClasses, historyPath, directory):
    directory.mkdir(parents=True)
    return {key: value for key, value in {
        "classes": classFigure(detection, directory), "matches": matchFigure(detection, directory),
        "precisionRecall": precisionRecallFigure(detection, pointClasses, directory),
        "confusion": confusionFigure(detection, directory), "points": pointFigure(points, config, directory),
        "history": historyFigure(historyPath, directory)}.items() if value is not None}
