# Court homography

`evaluateHomography` estimates, for every validation frame, the plane homography
between the image and a metric court model from predicted court landmarks. It
decides explicitly whether each homography is usable, projects athlete ground
points onto the court and measures the result against reference homographies
fitted to the annotated landmarks. It reads sealed evaluation archives only: it
runs no inference, trains nothing and never opens reserved test data.

## Court model

`configs/courtGeometry.yaml` defines a metric court in metres. The origin is
`upperCornerSideA`; x runs along the touchline toward Side B and y from the upper
touchline, farthest from the camera, to the lower touchline, nearest to it. Side
A/B is fixed court geometry and is unrelated to team A/B.

The default is 40 x 20 m. Regulation markings keep their distances when the
dimensions change: goalposts 1.5 m from the goal centre, penalty marks at 6 m and
10 m, the centre circle radius 3 m and substitution points 5 m and 10 m from the
halfway line on the lower touchline. Dimensions can be supplied in two ways:

- `--courtLength` and `--courtWidth` apply to every match (`commandLine`).
- `matchDimensions` in a copy of the configuration applies measured dimensions to
  one original match (`matchConfig`) and takes precedence.

Without either, results are labelled `nominalDefault`: metric errors then assume
that each real court measures 40 x 20 m. Wrong dimensions make the annotated
landmarks geometrically inconsistent, so references can be rejected. Lengths
must lie in 25-42 m and widths in 16-25 m. With 40 x 20 m the landmarks equal the
synthetic court references.

## Estimation

Each predicted landmark is the highest-confidence box centre of its class with
confidence at least `pointConfidence` (0.25). Boxes touching the image border are
counted because their centres can be shifted by truncation. Court Pose runs
archive the same detections, with boxes centred on the keypoints
([courtPose.md](courtPose.md)), so their evaluations are accepted unchanged.

With at most twenty landmarks, every four-landmark subset is examined. Subsets
containing three court points closer to collinear than `minTriangleAreaSquareMeters`
are skipped; many landmarks share a line (both goal lines, the halfway line, both
touchlines and the line through all four penalty marks). Each subset is fitted
exactly and scored by its inliers, reprojection errors of at most
`inlierThresholdDiagonal` (1% of the image diagonal), with ties broken by the
lowest truncated squared error. Fits that put their own points behind the camera
are not models. The best inlier set is refitted by least squares with OpenCV.
The search is exhaustive and deterministic, so no random seed is involved.

Gates then produce reason codes:

| Reason | Meaning |
| --- | --- |
| `insufficientPoints` | Fewer than four landmarks. |
| `degenerateCourtConfiguration` | No four landmarks without three collinear court points. |
| `noConsistentSubset` | Every exact fit places its own points behind the camera. |
| `inlierRmseHigh` | Inlier RMSE above `maxInlierRmseDiagonal` (0.5% of the diagonal). |
| `insufficientImageSpread` | Inliers cover less than `minImageHullFraction` of the image. |
| `pointsBehindCamera` | An inlier projects with nonpositive homogeneous scale. |
| `orientationReversed` | Negative Jacobian determinant: Side A/B or upper/lower labels are mirrored. |
| `farNearInverted` | Moving toward the camera on the floor moves up in the image. |

The orientation gates matter because a mirrored labelling still fits with almost
no residual. A frame is `verified` with at least five inliers and no reason,
`minimal` with exactly four inliers and no reason, and `rejected` otherwise. A
minimal homography interpolates its four points exactly, so it carries no internal
evidence of correctness; treat it as unverified.

The reference homography fits every annotated landmark with the same gates, so
annotation inconsistency is rejected rather than silently down-weighted. Its
leave-one-out residuals estimate how well the annotated points constrain it.

## Projection

`geometry.courtProjection` maps image points to the court and reports the local
scale in metres per pixel. Athlete positions use the bottom centre of the box,
which assumes the feet touch the floor. Positions outside the court plus
`courtMarginMeters` are flagged, never clipped. Ball positions are not projected
because the ball height is unknown.

## Metrics

Frames are grouped by reference status (`verified`, `minimal`). Rates divide by
all frames of a cohort, so rejected predictions count as failures.

- Availability: status and reason counts per match.
- Court-plane error: the `gridStepMeters` grid of the court visible under the
  reference is mapped back through the predicted homography. A frame's p90 uses
  the nearest rank and counts points behind the camera as failures.
- False valid: an accepted frame whose p90 exceeds `falseValidP90Meters`.
- Visible IoU: agreement of the court area visible under both homographies.
- Landmark transfer: every annotated landmark projected by the predicted
  homography, including landmarks the detector missed.
- Athlete error, decomposed: annotated boxes with the predicted homography
  (court only), matched predicted boxes with the reference (detection only) and
  matched predicted boxes with the predicted homography (end to end). Boxes are
  matched with the existing class-agnostic IoU >= 0.50 rule.
- Status split and a verified-only acceptance policy.
- Confidence sensitivity at `diagnosticConfidences`, which is diagnostic only.

Thresholds in `configs/evaluation/homography.yaml` were fixed before results were
inspected. Validation frames also selected the court checkpoint, so tuning them on
these results would reuse the reporting set.

## Run

```text
python -m futsal evaluateHomography --courtEvaluationDir <courtEvaluation> --playerEvaluationDir <athleteEvaluation> --mediaStoreDir data/mediaStore --outputDir outputs/reports/homography/<name>
```

`--courtEvaluationDir` is a completed validation evaluation with one stage that
predicts all twenty landmarks: an independent court model or the unified model.
`--playerEvaluationDir` is optional; any completed stage with athlete classes is
accepted, and its records must use the same images as the court records.
`--mediaStoreDir` is optional and adds overlay examples. The input contract and
the hashes of the consumed references and predictions are verified first.

The output is a sealed evaluation (`kind: homographyEvaluation`) with
`evidence/frames.jsonl`, `evidence/athletes.jsonl`, `evidence/summary.json`, the
code snapshot and optional `evidence/examples/`. `report.md` is rendered at once
and can be regenerated with `reportRun --evaluationDir`.

## Limitations

Lens distortion is not modelled. A single checkpoint and seed establish no
significance. Frames whose annotated points cover a small region, typical of
close cameras, give weakly constrained references even when verified; check their
leave-one-out residuals before relying on their metric errors.
