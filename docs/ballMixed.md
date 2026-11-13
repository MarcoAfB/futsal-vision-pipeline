# Ball detection with public and own training images

Mixed ball training has one class, ball at ID 0, and one training stage. Every
selected YOLO26 size starts from its matching official pretrained weights and
uses public and own training records together from the first epoch. There is no
public-then-own schedule. The existing own-only ballOnly recipe/profile and the
object/player curricula remain available with their previous identities.

The default campaign trains n/s/m/l/x independently in that order: seed 42,
960px, FP32, SGD, nominal batch 64, maximum 100 epochs, lr0 0.001, patience 20 and
disabled augmentations. Existing remaining ball parameters are retained. Each
stage requests batch 8, with explicit 8/4/2/1 attempts only after a verified memory
failure. Ambiguous CUDA errors do not authorize batch reduction.

## Source mixture and eligibility

| Training source | Eligible images |
| --- | ---: |
| Own training matches | 2553 |
| Futsal4400 | 2763 |
| SmartFutsal | 636 |
| FutsalMOT | 398 |
| FutsalAnaliza | 900 |
| FutsalDetect | 691 |
| Total | 7941 |

These are audit values for the current native snapshot, not application constants.
The sealed manifest/contract are authoritative. Public images account for about
68% of the natural pool; own images account for about 32%. Each eligible record
is used once per epoch, shuffled, without replacement, oversampling or dropping
the final partial batch. There is no requirement that each batch contain both
origins; batch 1 necessarily holds a single image.

The current training pool contains 7012 positive and 929 valid negative images.
Validation stays entirely own: 541 images, 527 positives and 14 valid negatives,
with the same target geometry/membership as the previous own-only ball task.
Only own training matches enter training. Public sources remain training-only;
reserved test images and labels are never exported or evaluated.

`configs/ballMixed.yaml` declares own and all five reviewed public sources.
`configs/training/ballMixed.yaml` is the corresponding scientific profile.
The optional ignoredSourceClasses field explicitly ignores public player,
outfieldPlayer and goalkeeper semantics for this task. Original observations,
annotation indices and provenance are retained. Existing tasks without that
field keep their prior unresolved-athlete guard; unknown/duplicate/conflicting
ignore declarations are refused. Unmapped/malformed/missing labels, unreadable
images, invalid ball geometry and multiple ball annotations stay quarantined.

A valid nonempty source label containing only ignored athletes can represent a
negative ball image under the recorded source-ontology completeness review.
Missing or empty public label files are not silently converted into negatives.
The inactive unified25 coverage, original canonical eligibility, source evidence
and frozen match split remain unchanged. Generic projection remains own-only.

## Prepare and transport

Use the native curriculum that retains public records, rather than the own-only
allTasks projection, as the derivation source. Register existing image providers
once and keep originals immutable. From the repository root:

```bash
.venv/bin/python scripts/runBallTraining.py prepare \
  --sourceDatasetDir ../futsalWork/data/prepared/curriculumV1 \
  --mediaStoreDir ../futsalWork/data/mediaStore \
  --planDir ../futsalWork/data/plans/ballMixedV1 \
  --datasetDir ../futsalWork/data/views/ballMixedV1 \
  --dataRoot ../futsalWork/data/raw
```

Use actual existing paths; Windows uses `.venv/Scripts/python.exe`. Both output
directories must be new. The derived native collection is registered explicitly
before preparing the view. Already registered images are reused. dataRoot is
optional if all needed records are registered; otherwise public images bind to
verified original bytes through existing raw aliases. Missing own development
frames can be extracted once if necessary; reserved test extraction is refused.

The local audit identified one additional public image to bind, already present
with the expected native hash, and no missing own image. Receiver inventory
determines the actual server delta; a complete raw copy is unnecessary:

```bash
# On the receiver, export and then copy this small inventory to the sender:
.venv/bin/python scripts/sharedDataTransfer.py inventory \
  --storeDir ../futsalWork/data/mediaStore \
  --output ../futsalWork/transfers/ballReceiverInventory.json
# On the sender with a prepared mixed view:
python scripts/sharedDataTransfer.py pack \
  --datasetDir data/views/ballMixedV1 --storeDir data/mediaStore \
  --receiverInventory incoming/ballReceiverInventory.json \
  --archive outgoing/ballMixedV1.tar.gz --receipt outgoing/ballMixedV1.receipt.json
# On the receiver after transferring the archive/receipt:
.venv/bin/python scripts/sharedDataTransfer.py unpack \
  --storeDir ../futsalWork/data/mediaStore \
  --archive incoming/ballMixedV1.tar.gz --receipt incoming/ballMixedV1.receipt.json \
  --outputDir ../futsalWork/data/received/ballMixedV1
export FUTSAL_BALL_VIEW=/home/user_marco/futsalWork/data/received/ballMixedV1/view
```

Keep the received directory's view and assets immutable as registered providers.

## Run, monitor and resume

Keep active campaigns' checkouts and environments frozen. Deploy the committed
mixed-ball implementation and run when GPU 0 is available. Reuse existing
futsalWork, its media store and per-size official weight sidecars; the launcher
does not download packages or weights. The Python environment must import this
checkout. Use the Bash wrapper for preparation/reuse and all-size execution:

```bash
bash scripts/runBallMixed.sh
```

FUTSAL_CODE and FUTSAL_WORK override the existing server defaults. Optional
FUTSAL_CURRICULUM_DATASET, FUTSAL_RAW_DATA_ROOT and FUTSAL_BALL_VIEW select
existing input locations. A missing raw provider can instead be supplied by the
incremental transfer workflow above. The wrapper can reuse a matching native
plan after an interrupted preparation, without replacing prior outputs.

Campaigns are named ballMixedAllSizesSeed42_<UTCtimestamp>_<pid>. Runs/reports
live under futsalWork/runs|reports/<campaign>, with fresh
yolo26{size}RealAttemptNN directories. Environment, training and reporting workers
execute sequentially; follow each announced log in a separate terminal with
`tail -n 80 -F <worker-log>`. GPU use can be inspected with `nvidia-smi`.
The dedicated Python launcher also exposes prepare/train with modelSizes,
campaignDir, weightsRoot, evaluationConfig and resume options.

On OOM, the entire single stage restarts at epoch zero from the same official
weights, with a fresh optimizer and next smaller batch. Partial checkpoints and
optimizer state are not continued. Batch 1 exhaustion advances to the next size.
Interruption stops the queue; explicit campaign resume restarts an incomplete
stage at the same batch. Set FUTSAL_CAMPAIGN_DIR to the printed existing path and
rerun the wrapper for resume. Identity/lock checks and attempt preservation are
shared with existing campaigns. A changed code/data/weights/profile/mode cannot
resume another campaign. Completed training with report-only failure is never
retrained; a completed sealed evaluation can be rendered without inference.

## Audit and interpretation

Native loader audits require shuffled RandomSampler without replacement or
dropped batches. Each completed training epoch checks actual record IDs against
the complete immutable selection. exposure.jsonl records verified coverage,
order hash and source composition, including the last partial batch. The run
records the sampling policy and verified epoch count. Reports archive that
evidence and show origin/source/positive/negative counts and requested/effective
batches. Failure reports keep sealed selection/execution metadata without
inventing research scores. reportRun rendering stays inference-free.

The best checkpoint is chosen only on own validation. More sources are a
testable hypothesis, not evidence of higher accuracy. The old own-only runs stay
available as historical references. Differences in effective batch, code and
optimizer updates must be explicit: 100 epochs of 7941 images consume more
training data than 100 epochs of 2553. No automatic strategy comparison, tracking,
public-then-own training or final-test evaluation is introduced here.
