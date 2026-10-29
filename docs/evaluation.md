# Validation research reports

Research evaluation is separate from training. Reports identify their checkpoint,
dataset contract, exact record IDs, protocol and purpose. The technical smoke is
not a scientific model comparison. Test images and labels remain reserved.

`configs/evaluation/researchValidation.yaml` freezes confidence floor 0.001,
operational confidence 0.25, operational IoU 0.50, AP IoU thresholds 0.50:0.95,
and point tolerances 0.25%, 0.5%, 1%, 2% and 5% of the original image diagonal.
The primary point tolerance is 1%. These defaults are diagnostic thresholds, not
automatically calibrated or model-quality acceptance thresholds.

`runValidation` uses exactly the saved training validation selection, verified
against the prepared manifest. `fullValidation` uses all eligible own validation
records for each task. The two selections produce separate evaluations. They do
not change task quarantine or the original-match split.

References retain original image dimensions, derived pixel boxes, source annotation
indices and native point coordinates. A prediction JSONL contains one complete
record per expected image, including empty detection lists. Missing, duplicate,
unexpected, incorrectly sized or incorrectly classified records fail validation.
Zero-area predicted boxes remain false-positive candidates rather than disappearing.
