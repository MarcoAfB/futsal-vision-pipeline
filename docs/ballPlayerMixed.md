# Ball and generic player detection with public and own training images

Mixed ball-player training has two classes, ball at ID 0 and player at ID 1, and
one training stage. The generic player includes outfield players and goalkeepers
of both teams; referees are never targets. Every selected YOLO26 size starts from
its matching official pretrained weights and uses public and own training records
together from the first epoch. There is no public-then-own schedule. The mixed
ball and mixed player workflows, the own-only genericObjects task and the object
curriculum keep their previous identities.

The default campaign trains n/s/m/l/x independently in that order with the mixed
ball recipe: seed 42, 960px, FP32, SGD, nominal batch 64, maximum 100 epochs, lr0
0.001, patience 20 and disabled augmentations. Each stage requests batch 8, with
explicit 8/4/2/1 attempts only after a verified memory failure. Ambiguous CUDA
errors do not authorize batch reduction.

## Source mixture and eligibility

| Training source | Eligible images | Ball-free images |
| --- | ---: | ---: |
| Own training matches | 2553 | 264 |
| Futsal4400 | 2762 | 1 |
| SmartFutsal | 636 | 92 |
| FutsalMOT | 398 | 38 |
| FutsalAnaliza | 900 | 307 |
| FutsalDetect | 691 | 227 |
| Total | 7940 | 929 |

These are audit values for the current native snapshot, not application constants.
The sealed manifest/contract are authoritative. Public images account for about
68% of the natural pool; own images account for about 32%. Each eligible record
is used once per epoch, shuffled, without replacement, oversampling or dropping
the final partial batch.

`configs/ballPlayerMixed.yaml` maps ball to ball and the public player,
outfieldPlayer and goalkeeper semantics and the four own team/role classes to
player. `configs/training/ballPlayerMixed.yaml` is the corresponding scientific
profile. An image is quarantined when either class has invalid geometry, when it
has more than one ball, or when labels are unmapped, malformed or missing; empty
public label files and unreadable images stay quarantined. Images with players
and no ball are valid, ball-free training images. Every eligible image has at
least one player, so the pool contains no empty image; the report's negative-image
column therefore shows 0.

Validation stays entirely own: the same 541 images and ball/athlete targets as the
mixed ball task and the own-only genericObjects task (527 ball and 3055 player
references). Reserved test images and labels are never exported or evaluated.

## Prepare and transport

```bash
.venv/bin/python scripts/runBallPlayerTraining.py prepare \
  --sourceDatasetDir ../futsalWork/data/prepared/curriculumV1 \
  --mediaStoreDir ../futsalWork/data/mediaStore \
  --planDir ../futsalWork/data/plans/ballPlayerMixedV1 \
  --datasetDir ../futsalWork/data/views/ballPlayerMixedV1 \
  --dataRoot ../futsalWork/data/raw
```

Use actual existing paths; Windows uses `.venv/Scripts/python.exe`. Both output
directories must be new. The mixed ball and player views already registered these
images, so no image is copied; use the receiver inventory and incremental transfer
described in [sharedData.md](sharedData.md) for any server delta, then set
FUTSAL_BALL_PLAYER_MIXED_VIEW to the received view.

## Run, monitor and resume

Keep active campaigns' checkouts and environments frozen. Run when GPU 0 is free:

```bash
bash scripts/runBallPlayerMixed.sh
```

FUTSAL_CODE, FUTSAL_WORK, FUTSAL_CURRICULUM_DATASET, FUTSAL_RAW_DATA_ROOT and
FUTSAL_BALL_PLAYER_MIXED_VIEW select existing locations. Campaigns are named
ballPlayerMixedAllSizesSeed42_<UTCtimestamp>_<pid>. To train selected sizes only,
call `scripts/runBallPlayerTraining.py train --modelSizes ...` directly. To resume,
set FUTSAL_CAMPAIGN_DIR to the printed path and rerun the wrapper. OOM,
interruption, identity and report-only recovery behave exactly as in the
[mixed ball workflow](ballMixed.md).

## Audit and interpretation

Loader audits, per-epoch exposure.jsonl coverage and report composition are the
same as for mixed ball. The best checkpoint is chosen only on own validation, by
native box AP over both classes. Ball boxes are about 11% of all training targets,
so the selected epoch is shared by both classes; the report keeps ball and athlete
scores separate.

The research question is whether one two-class model can replace the separate
mixed ball and mixed player models. Compare on the same 541 validation frames, with
the same evaluation protocol: ball AP50-95/F1 against the mixed ball model, and
class-collapsed athlete precision/recall/F1 against the mixed player model of the
same size, restricted to these frames. Decide the margin treated as "no worse"
before training. A single seed cannot separate differences of about one point from
seed variation. Scores are generic-athlete scores and are not comparable with
detailed team/role class AP. This model cannot be used by the current ball tracking
path, which accepts only a single-class ballOnly checkpoint.
