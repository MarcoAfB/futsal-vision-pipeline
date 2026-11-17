# Team and role assignment

The generic player detector finds athletes without team or role. This component
assigns each of its detections one of the four athlete classes,
`outfieldPlayerTeamA`, `outfieldPlayerTeamB`, `goalkeeperTeamA` and
`goalkeeperTeamB`, and scores the result against the detailed validation labels.
It reads sealed evaluation archives only, runs no detector and never reads the
reserved test data. It uses no tracking: every detection is classified on its own,
from a model of its match fitted once.

Team A/B identifies a team within one original match only. It follows the
initial-side convention: team A defends Side A in the clip. Court Side A/B is
fixed geometry. A single frame does not show which team is "A", which is why a
detector that predicts the four classes directly has to guess.

## Two steps

```text
extractTeamFeatures  sealed playerOnly evaluation + media store -> kind teamFeatures
evaluateTeams        teamFeatures + homographyEvaluation + detailed reference -> kind teamEvaluation
```

`extractTeamFeatures` needs the images and, for SigLIP, a GPU. It runs on the
server. `evaluateTeams` needs only NumPy, scikit-learn and UMAP and runs where the
sealed archives are. The player evaluation may be any sealed validation stage
with a `player` class, such as player-only or ball-player; only its `player`
detections are used. Only the court homographies of the homography evaluation
are used, so it may come from another player evaluation, but every shared frame
must have the same image hash.

### Features

For every generic `player` detection at or above `detectionConfidence`, a torso
window is cropped (`torsoTop` to `torsoBottom` of the box height, without
`sideMargin` of the width on each side). A crop whose box overlaps another
detection above `overlapIouMax` is marked unclean: it is classified but does not
fit the clusters. Each crop gets one or both of:

- `siglip`: the L2-normalized pooled output of the frozen SigLIP vision encoder
  named in the configuration, at a fixed revision.
- `colorHistogram`: a square-rooted HSV histogram, a cheap baseline that also
  serves the software tests.

Evidence: `crops.jsonl` (one row per detection), `frames.jsonl` (every frame of
the player evaluation, including frames without detections) and `embeddings.npz`.

### Assignment

Per original match:

1. Clean crops are reduced with UMAP (cosine, fixed seed) and clustered with
   K-means into `clusterCount` groups; every crop of the match is then assigned
   to a group.
2. The two largest groups are the outfield teams.
3. A smaller group becomes a goalkeeper of one side when it holds at most
   `goalkeeperMaxShare` of the crops and either
   - at least `minEvidence` members are projected by an accepted homography and
     at least `goalkeeperZoneMin` of them lie within `goalRadiusMeters` of one goal
     centre (`goalZone`), or
   - too few members are projected, and the side-specific landmarks detected in
     their frames agree on one side in at least `goalkeeperSideMin` of the votes
     (`sideVote`).
   One goalkeeper group per side is kept, the one with the stronger evidence.
4. Other groups join the nearest team centre (`mergedNearestTeam`).
5. In each frame at most one goalkeeper per side is kept; extra ones join the
   nearest team (`frameGoalkeeperLimit`).

Names: a goalkeeper is named after the side of the goal it defends. The outfield
team whose projected members have the lower mean court x is team A (`meanX`).
Without `minEvidence` projected members per team the outfield names are arbitrary
(`none`). Projections use the bottom centre of the box and only frames whose
homography status is listed in `homographyStatuses` (default `verified`).

### Scoring

Predictions are matched to the detailed references class-agnostically at
`matchIou`. The summary reports role accuracy (goalkeeper or outfield), team
accuracy and class accuracy over matched athletes, plus per-class operational
precision, recall and F1 and the confusion matrix, overall and by match.

Each score appears twice:

- `named`: as predicted.
- `bestSwap`: A and B exchanged jointly for outfield players and goalkeepers in
  each match where that increases correct classes. It reads the labels, so it is
  a diagnostic of the naming rule, not a deployable result.

The four-class detector of the detailed reference evaluation is scored the same
way as a baseline. Its scores therefore separate localization and role errors
from the arbitrary A/B convention.

## Commands

Install `--extra trainingGpu --extra teamFeatures` on the server and
`--extra teamClustering --extra reporting` where the evaluation runs. The SigLIP
files are downloaded once with network access:

```bash
HF_HUB_OFFLINE=0 .venv/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download('google/siglip-base-patch16-224', revision='<siglipRevision>', \
allow_patterns=['config.json', 'model.safetensors', 'preprocessor_config.json'])"
```

On the server, from a clean checkout:

```bash
FUTSAL_PLAYER_EVALUATION=<playerOnly runValidation directory> bash scripts/runTeamFeatures.sh
```

Then, after the result transfer:

```bash
uv run python -m futsal evaluateTeams --featuresDir <teamFeatures> \
  --homographyEvaluationDir <homographyEvaluation> --referenceEvaluationDir <detailed runValidation> \
  --outputDir <new directory>
```

`reportRun --evaluationDir` regenerates the Markdown report from the sealed
evidence.

## Limitations

- Match 11 is the hardest validation match. Its broadcast looks more amateur than
  the others, and the researcher deliberately placed it in validation to probe the
  limits of the models. Weak results there (for example no usable court homography
  and goalkeepers that cannot be separated by appearance) describe those limits;
  they are not a defect of the sample. Report it separately and read pooled
  figures with that in mind. This is the researcher's description of the source,
  not a quantity measured by this component.
- Thresholds in `configs/evaluation/teamAssignment.yaml` were fixed before any
  validation result; adjusting them on validation frames would bias the result.
  Tune on training matches if needed.
- A clip that crosses a change of ends breaks the side convention.
- Goalkeeper groups need enough frames of that goalkeeper; a goalkeeper seen in
  very few frames is merged into a team.
- Referees are not detector targets; any detected referee is assigned to a group
  and counts as a false positive.
