from collections import Counter

from futsal.evaluation.pointMetrics import scorePoints


def pointSelectionScore(references, predictions, names, selection):
    """Score one validation epoch against native points, as the sealed evaluation does."""
    byId = {row["recordId"]: row for row in predictions}
    if len(byId) != len(predictions) or set(byId) != {row["recordId"] for row in references}:
        raise ValueError("Point selection predictions must cover each validation record exactly once.")
    tolerance = selection["tolerance"]
    config = {"operationalConfidence": selection["confidence"], "primaryPointTolerance": tolerance,
              "pointTolerances": [tolerance]}
    scores = scorePoints(references, [byId[row["recordId"]] for row in references], names, set(names), config)
    key = str(tolerance)
    rates = [row["successRates"][key] for row in scores["perClass"] if row["support"]]
    if not rates:
        raise ValueError("Point selection requires validation classes with native points.")
    correct = Counter(row["recordId"] for row in scores["errors"] if row["distanceDiagonal"] is not None
                      and row["distanceDiagonal"] <= tolerance)
    return {"macro": sum(rates) / len(rates), "micro": scores["successRates"][key],
            "framesWithFourCorrect": sum(count >= 4 for count in correct.values()),
            "frames": len(references)}


def pointSelectionValidator(dataloader, saveDir, args, callbacks, references, names, selection, recordOf):
    from ultralytics.models.yolo.detect import DetectionValidator

    class PointSelectionValidator(DetectionValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            self.pointPredictions = []

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for index, pred in enumerate(preds):
                pbatch = self._prepare_batch(index, batch)
                scaled = self.scale_preds(self._prepare_pred(pred), pbatch)
                detections = [{"classId": int(classId), "confidence": float(confidence), "box": box}
                              for classId, confidence, box in zip(scaled["cls"].tolist(), scaled["conf"].tolist(),
                                                                  scaled["bboxes"].tolist(), strict=True)]
                self.pointPredictions.append({"recordId": recordOf(pbatch["im_file"]), "detections": detections})

        def pointScores(self):
            return pointSelectionScore(references, self.pointPredictions, names, selection)

    return PointSelectionValidator(dataloader, save_dir=saveDir, args=args, _callbacks=callbacks)
