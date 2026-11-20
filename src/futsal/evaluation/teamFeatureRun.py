import math
import traceback
from importlib.metadata import version
from pathlib import Path

import numpy as np

from futsal.datasets.mediaStore import protectMediaOutput, readBindings
from futsal.evaluation.evaluationData import validateResearchOutput, writeJsonLines
from futsal.evaluation.evaluationRun import saveEvaluation, sealEvaluation
from futsal.evaluation.homographyRun import sealedStage
from futsal.evaluation.trackingRun import sealedTrackingCondition
from futsal.inventory.catalog import readYaml
from futsal.reporting.homographyFigures import loadImage
from futsal.teams.appearanceFeatures import colorHistogram, loadSiglip, siglipEmbeddings
from futsal.teams.athleteCrops import cropImage, detectionCrops
from futsal.training.trainingRun import snapshotCode, utcNow

featureKinds = ("siglip", "colorHistogram")
fractionFields = {"detectionConfidence", "torsoTop", "torsoBottom", "sideMargin", "overlapIouMax"}
integerFields = {"minCropPixels", "batchSize"}
textFields = {"siglipModel", "siglipRevision"}


def loadTeamFeatureConfig(path):
    config, evidence = readYaml(Path(path))
    if set(config) != {"schemaVersion", "featureKinds", "histogramBins"} | fractionFields | integerFields | textFields:
        raise ValueError("Unexpected or missing team feature configuration fields.")
    if config["schemaVersion"] != 1:
        raise ValueError("Only team feature schemaVersion 1 is supported.")
    for key in fractionFields:
        if type(config[key]) not in {int, float} or not math.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be finite and in [0, 1].")
    if config["torsoTop"] >= config["torsoBottom"] or config["sideMargin"] >= 0.5:
        raise ValueError("The torso window must have positive height and width.")
    if any(type(config[key]) is not int or config[key] < 1 for key in integerFields):
        raise ValueError("minCropPixels and batchSize must be positive integers.")
    if any(not isinstance(config[key], str) or not config[key] for key in textFields):
        raise ValueError("siglipModel and siglipRevision must be nonempty strings.")
    kinds = config["featureKinds"]
    if not isinstance(kinds, list) or not kinds or len(set(kinds)) != len(kinds) or not set(kinds) <= set(featureKinds):
        raise ValueError(f"featureKinds must be distinct values from {featureKinds}.")
    bins = config["histogramBins"]
    if not isinstance(bins, list) or len(bins) != 3 or any(type(value) is not int or value < 1 for value in bins):
        raise ValueError("histogramBins must list three positive integers for hue, saturation and value.")
    return config, evidence


def collectCrops(references, predictions, mediaStoreDir, config, progress):
    bindings = readBindings(mediaStoreDir)
    rows, crops = [], []
    for index, (reference, prediction) in enumerate(zip(references, predictions, strict=True), start=1):
        image = loadImage(mediaStoreDir, reference["imageSha256"], bindings)
        if [image.shape[1], image.shape[0]] != list(reference["dimensions"]):
            raise ValueError(f"Image dimensions differ from the sealed record {reference['recordId']}.")
        for item in detectionCrops(prediction, config):
            featureIndex = None
            if item["torsoBox"] is not None:
                featureIndex = len(crops)
                crops.append(cropImage(image, item["torsoBox"]).copy())
            rows.append({"recordId": reference["recordId"], "matchId": reference["matchId"],
                         "imageSha256": reference["imageSha256"], "featureIndex": featureIndex,
                         **({"segmentId": prediction["segmentId"]} if "segmentId" in prediction else {}), **item})
        if index % 100 == 0 or index == len(references):
            progress(f"Athlete crops: {index}/{len(references)} frames, {len(crops)} crops")
    return rows, crops


def playerSource(sourceDir, condition):
    """A sealed player evaluation, or one condition of a sealed tracking campaign when condition is given."""
    if condition is None:
        identity, references, predictions = sealedStage(
            sourceDir, lambda classes: "player" in classes, "Player evaluation")
        return "sealedPlayerValidation", {"player": identity}, references, predictions
    identity, references, predictions = sealedTrackingCondition(sourceDir, condition)
    if "player" not in identity["classes"]:
        raise ValueError("Tracking campaign: the tracked task has no player class.")
    return "sealedTrackingCondition", {"tracking": identity}, references, predictions


def extractTeamFeatures(sourceDir, mediaStoreDir, outputDir, configPath, *, condition=None, device="cpu",
                        progress=print):
    outputDir = validateResearchOutput(outputDir, [Path(sourceDir)], [Path(configPath)])
    protectMediaOutput(outputDir, mediaStoreDir)
    config, configEvidence = loadTeamFeatureConfig(configPath)
    selection, inputs, references, predictions = playerSource(sourceDir, condition)
    evidenceDir = outputDir / "evidence"
    evidenceDir.mkdir(parents=True)
    state = {"schemaVersion": 1, "kind": "teamFeatures", "status": "running", "startedAt": utcNow(),
             "purpose": "teamAppearanceFeatures", "selection": selection, "testEvaluated": False,
             "inputs": inputs, "configuration": config, "configurationEvidence": configEvidence, "device": device}
    saveEvaluation(outputDir, state)
    try:
        state["code"] = snapshotCode(evidenceDir)
        state["packages"] = {name: version(name) for name in ("numpy", "opencv-python-headless")}
        encoder = None
        if "siglip" in config["featureKinds"]:
            encoder, state["model"] = loadSiglip(config["siglipModel"], config["siglipRevision"], device)
        rows, crops = collectCrops(references, predictions, mediaStoreDir, config, progress)
        # Frames without detections still belong to the cohort; their references become misses.
        writeJsonLines(evidenceDir / "frames.jsonl", [{key: reference[key] for key in
                                                       ("recordId", "matchId", "imageSha256", "dimensions")}
                                                      for reference in references])
        features = {}
        if "colorHistogram" in config["featureKinds"]:
            features["colorHistogram"] = np.array([colorHistogram(crop, config["histogramBins"]) for crop in crops],
                                                  dtype=np.float32).reshape(len(crops), -1)
        if encoder is not None:
            progress(f"Embedding {len(crops)} crops with {config['siglipModel']} on {device}...")
            features["siglip"] = siglipEmbeddings(crops, encoder, config["batchSize"])
        writeJsonLines(evidenceDir / "crops.jsonl", rows)
        np.savez(evidenceDir / "embeddings.npz", **features)
        state.update(status="completed", frameCount=len(references), detectionCount=len(rows),
                     cropCount=len(crops), cleanCropCount=sum(row["clean"] for row in rows),
                     featureShapes={name: list(value.shape) for name, value in features.items()})
    except (Exception, KeyboardInterrupt) as error:
        (outputDir / "pipelineFailure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error={"type": type(error).__name__, "message": str(error)})
        sealEvaluation(outputDir, state)
        raise
    sealEvaluation(outputDir, state)
    return state
