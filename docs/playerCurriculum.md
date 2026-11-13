# Player-only public-to-own curriculum

The player curriculum trains one detector through 1/2/4 classes, with no ball
targets or outputs. The generic player includes both outfield players and
goalkeepers. The final four classes retain the recording's annotated team A/B
identity, separate from fixed court sides and universal club identities.

| Task | Ordered classes | Training sources | Max epochs | lr0 | Patience |
| --- | --- | --- | --- | --- | --- |
| genericPlayers | player | Futsal4400, SmartFutsal | 80 | 0.001 | 15 |
| rolePlayers | outfieldPlayer, goalkeeper | FutsalMOT, FutsalAnaliza, FutsalDetect | 60 | 0.0005 | 15 |
| detailedPlayers | outfieldPlayerTeamA, outfieldPlayerTeamB, goalkeeperTeamA, goalkeeperTeamB | Own training matches | 100 | 0.0002 | 20 |

Each size n/s/m/l/x starts independently from its official weights and transfers
the preceding stage's best checkpoint. Compatible tensors are preserved and new
class outputs are learned from labels, without reinterpreting old output indices.
Every stage starts a fresh optimizer. Seed 42, 960px, FP32, SGD, nbs64, disabled
augmentations and the remaining object-curriculum parameters are retained.
The final detailedPlayers best checkpoint is the resulting four-class model.

## Prepare the shared view

`configs/playerCurriculum.yaml` declares the three active tasks. unified25 stays
inactive canonical coverage: its original ball/court annotations and eligibility
preserve the frozen split. Only player targets are written for the active tasks.
`configs/training/yolo26PlayerCurriculum.yaml` is the matching training profile.
Existing playerOnly standalone training and object recipes stay available.

The derivative retains all native public/own records, source evidence and frozen
match assignments. Player eligibility is recomputed independently of ball-only
geometry failures. Relevant player, unknown-label, incomplete-scope and source
failures remain quarantined. Public data is training-only; all stages use own
validation. Reserved test images and labels are neither exported nor evaluated.

For the current native source snapshot, expected train counts are 3401/2012/2585
and validation counts 550/550/550, for 9648 task rows. The manifest and sealed
contract are authoritative; these audit values are not application constants.
The cohorts differ from the object curriculum's 3398/1989/2553 and 541 validation
records. Direct full-cohort comparisons would change both task definitions and
data membership. Stage scores also describe different ontologies.

From the repository root, using the existing registered store and raw inputs:

```bash
.venv/bin/python scripts/runObjectCurriculum.py prepare --curriculum players \
  --sourceDatasetDir ../futsalWork/data/prepared/curriculumV1 \
  --mediaStoreDir ../futsalWork/data/mediaStore \
  --planDir ../futsalWork/data/plans/playerCurriculumV1 \
  --datasetDir ../futsalWork/data/views/playerCurriculumV1 \
  --dataRoot ../futsalWork/data/raw
```

Use actual existing source/store paths; Windows uses `.venv/Scripts/python.exe`.
`--dataRoot` is optional when all needed images are already registered. It can
read a relocated raw sourceManifest and verify its aliases. Missing public
images are bound to their original bytes; already registered images are reused.
Missing own development frames, when necessary, are extracted once into the
store. No original source annotations/media or prior contracts are modified.
Both plan and view directories must be new. The exact derived collection is
registered before preparing the view; generic projection remains own-data-only.

The local preparation audit identified 26 additional public records to bind and
no missing own frames. These public files are present in local raw inputs with
verified hashes. Server availability is determined separately by its inventory.

## Transfer only absent assets

After the current server campaign finishes, export its existing verified media
inventory and move that small file to the sender:

```bash
.venv/bin/python scripts/sharedDataTransfer.py inventory \
  --storeDir ../futsalWork/data/mediaStore \
  --output ../futsalWork/transfers/playerReceiverInventory.json
```

On the sender, package the new view against that receiver inventory:

```bash
python scripts/sharedDataTransfer.py pack \
  --datasetDir data/views/playerCurriculumV1 --storeDir data/mediaStore \
  --receiverInventory incoming/playerReceiverInventory.json \
  --archive outgoing/playerCurriculumV1.tar.gz \
  --receipt outgoing/playerCurriculumV1.receipt.json
```

Transfer the archive/receipt once. On the server:

```bash
.venv/bin/python scripts/sharedDataTransfer.py unpack \
  --storeDir ../futsalWork/data/mediaStore \
  --archive incoming/playerCurriculumV1.tar.gz \
  --receipt incoming/playerCurriculumV1.receipt.json \
  --outputDir ../futsalWork/data/received/playerCurriculumV1
export FUTSAL_PLAYER_VIEW=/home/user_marco/futsalWork/data/received/playerCurriculumV1/view
```

Only absent image hashes travel. The received view remains under `view`, with
new asset providers under `assets`; keep the received directory immutable.
There is no need to provision or transfer the complete raw dataset to the server.

## Run, monitor and resume

Keep the active object campaign's checkout/environment frozen. Develop in a
separate checkout, then deploy the committed player implementation and run it
after that campaign finishes on GPU 0. Use an environment importing futsal from
the selected checkout. A changed code identity cannot resume an old campaign.

The supplied Bash wrapper reuses the existing workspace, store and official
weight sidecars. It verifies an existing player view, or prepares it when the
required providers/raw inputs are available. Its defaults match the existing
server layout; FUTSAL_CODE, FUTSAL_WORK, FUTSAL_CURRICULUM_DATASET,
FUTSAL_RAW_DATA_ROOT and FUTSAL_PLAYER_VIEW can select existing alternatives.

```bash
bash scripts/runPlayerCurriculum.sh
```

The default campaign is playerCurriculumAllSizesSeed42_<UTCtimestamp>_<pid>.
Runs/reports live under futsalWork/runs|reports/<campaign>, with fresh
yolo26{size}RealAttemptNN directories. Environment preflight, training and report
workers run sequentially; stdout/stderr are written to each announced worker log.
Use `tail -n 80 -F <worker-log>` in a separate terminal to follow stage/epoch
progress. GPU memory can be inspected independently with `nvidia-smi`.

Every stage requests batch8. Only a definite allocation failure allows 8/4/2/1
for that failed stage, reusing the verified completed prefix. The stage restarts
at epoch zero with a fresh optimizer and preceding best weights; the next stage
starts at 8 again. Native silent reductions are blocked. Batch1 exhaustion marks
the size failed and advances to the next size. Interruption stops the queue.

For resume, set FUTSAL_CAMPAIGN_DIR to the printed existing campaign path and
rerun the wrapper with identical code, inputs and weights. A matching manifest
selects `--resume`. Locks and unsealed interrupted workers require the same
inspection rules as the object campaign. Completed training is never repeated
because only its report failed; completed sealed evaluations can be rendered
without inference. Reports retain classes/support/sources, requested/effective
batches, checkpoint hashes and reuse ancestry.

For direct CLI use, `train --curriculum players` defaults to all five sizes and
the player profile; `--modelSizes n` can select fewer sizes. Omitting
`--curriculum` preserves the objects variant. Variant/view/profile mismatch is
refused and the selected variant participates in campaign identity.

Local fixture tests establish engineering behavior. Actual GPU fit, completed
scientific training and model quality require execution evidence. Tracking,
homography, automatic comparisons and reserved-test evaluation are separate work.
