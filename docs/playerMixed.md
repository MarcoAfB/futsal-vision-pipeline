# Generic player detection with public and own training images

Mixed player training has one class, player at ID 0, and one training stage. The
generic player includes outfield players and goalkeepers of both teams; referees
are never targets. Every selected YOLO26 size starts from its matching official
pretrained weights and uses public and own training records together from the
first epoch. There is no public-then-own schedule. The own-only playerOnly task,
the player curriculum and the mixed ball workflow keep their previous identities.

The default campaign trains n/s/m/l/x independently in that order with the mixed
ball recipe: seed 42, 960px, FP32, SGD, nominal batch 64, maximum 100 epochs, lr0
0.001, patience 20 and disabled augmentations. Each stage requests batch 8, with
explicit 8/4/2/1 attempts only after a verified memory failure. Ambiguous CUDA
errors do not authorize batch reduction.

## Source mixture and eligibility

| Training source | Eligible images |
| --- | ---: |
| Own training matches | 2585 |
| Futsal4400 | 2764 |
| SmartFutsal | 637 |
| FutsalMOT | 404 |
| FutsalAnaliza | 902 |
| FutsalDetect | 706 |
| Total | 7998 |

These are audit values for the current native snapshot, not application constants.
The sealed manifest/contract are authoritative. Public images account for about
68% of the natural pool; own images account for about 32%. Each eligible record
is used once per epoch, shuffled, without replacement, oversampling or dropping
the final partial batch. There is no requirement that each batch contain both
origins; batch 1 necessarily holds a single image.

`configs/playerMixed.yaml` maps the public player, outfieldPlayer and goalkeeper
semantics and the four own team/role classes to player.
`configs/training/playerMixed.yaml` is the corresponding scientific profile.
Ball and court labels are outside this task: they are not targets and their
geometry cannot quarantine an image. Unmapped/malformed/missing labels, unreadable
images, empty public label files and invalid athlete geometry stay quarantined.
A valid label containing only ball or referee annotations would be a negative
player image under the recorded source-ontology completeness review; the current
pool has none, so every training image contains at least one player.

Validation stays entirely own: 550 images with the same membership and targets as
the own-only playerOnly task. Own recordings contain no generic player boxes, so
mapping that semantic changes no own record. Only own training matches enter
training. Public sources remain training-only; reserved test images and labels
are never exported or evaluated. The inactive unified25 coverage, original
canonical eligibility, source evidence and frozen match split remain unchanged.

## Prepare and transport

Use the native curriculum that retains public records, rather than the own-only
allTasks projection, as the derivation source. From the repository root:

```bash
.venv/bin/python scripts/runPlayerTraining.py prepare \
  --sourceDatasetDir ../futsalWork/data/prepared/curriculumV1 \
  --mediaStoreDir ../futsalWork/data/mediaStore \
  --planDir ../futsalWork/data/plans/playerMixedV1 \
  --datasetDir ../futsalWork/data/views/playerMixedV1 \
  --dataRoot ../futsalWork/data/raw
```

Use actual existing paths; Windows uses `.venv/Scripts/python.exe`. Both output
directories must be new. Already registered images are reused; dataRoot is
optional if all needed records are registered. Receiver inventory determines the
actual server delta; a complete raw copy is unnecessary:

```bash
# On the receiver, export and then copy this small inventory to the sender:
.venv/bin/python scripts/sharedDataTransfer.py inventory \
  --storeDir ../futsalWork/data/mediaStore \
  --output ../futsalWork/transfers/playerReceiverInventory.json
# On the sender with a prepared mixed view:
python scripts/sharedDataTransfer.py pack \
  --datasetDir data/views/playerMixedV1 --storeDir data/mediaStore \
  --receiverInventory incoming/playerReceiverInventory.json \
  --archive outgoing/playerMixedV1.tar.gz --receipt outgoing/playerMixedV1.receipt.json
# On the receiver after transferring the archive/receipt:
.venv/bin/python scripts/sharedDataTransfer.py unpack \
  --storeDir ../futsalWork/data/mediaStore \
  --archive incoming/playerMixedV1.tar.gz --receipt incoming/playerMixedV1.receipt.json \
  --outputDir ../futsalWork/data/received/playerMixedV1
export FUTSAL_PLAYER_MIXED_VIEW=/home/user_marco/futsalWork/data/received/playerMixedV1/view
```

Keep the received directory's view and assets immutable as registered providers.

## Run, monitor and resume

Keep active campaigns' checkouts and environments frozen. Deploy the committed
implementation and run when GPU 0 is available. The wrapper reuses existing
futsalWork, its media store and per-size official weight sidecars; it does not
download packages or weights:

```bash
bash scripts/runPlayerMixed.sh
```

FUTSAL_CODE and FUTSAL_WORK override the existing server defaults. Optional
FUTSAL_CURRICULUM_DATASET, FUTSAL_RAW_DATA_ROOT and FUTSAL_PLAYER_MIXED_VIEW select
existing input locations. FUTSAL_PLAYER_VIEW belongs to the player curriculum and
is not read here. The wrapper can reuse a matching native plan after an
interrupted preparation, without replacing prior outputs.

Campaigns are named playerMixedAllSizesSeed42_<UTCtimestamp>_<pid>. Runs/reports
live under futsalWork/runs|reports/<campaign>. Follow each announced worker log
with `tail -n 80 -F <worker-log>`. To resume, set FUTSAL_CAMPAIGN_DIR to the
printed existing path and rerun the wrapper. OOM restarts, interruption,
identity/lock checks and report-only recovery behave exactly as in the
[mixed ball workflow](ballMixed.md). A player view cannot be launched as a ball
campaign, or the reverse.

## Audit and interpretation

Loader audits, per-epoch exposure.jsonl coverage and report composition are the
same as for mixed ball. The best checkpoint is chosen only on own validation.
Scores are generic-athlete scores: they are not comparable with detailed team/role
class AP. More sources are a testable hypothesis, not evidence of higher accuracy;
100 epochs of 7998 images consume more training data than 100 epochs of 2585.
Differences in effective batch, code and optimizer updates must be explicit. No
automatic strategy comparison, tracking, team assignment or final-test evaluation
is introduced here.
