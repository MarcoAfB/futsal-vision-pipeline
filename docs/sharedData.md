# Shared media and task views

Register immutable images once and derive lightweight task views. Images stay in
their existing verified provider directories. A view contains native provenance,
eligibility, task labels, image references and a sealed contract, without image
copies. Training and evaluation with `--mediaStoreDir` read these assets directly.
Old prepared bundles, completed runs and evaluation seals remain unchanged.

## Register existing data

Use the existing environment and work directory. The following example reuses an
existing server curriculum; it does not provision an environment or rerun training.
Run from the updated checkout, using its existing interpreter or execution wrapper.

```bash
workRoot=/home/user_marco/futsalWork
python -m futsal registerMedia \
  --storeDir "$workRoot/data/mediaStore" \
  --sourceDir "$workRoot/data/prepared/curriculumV1"
```

Registration performs a complete integrity check and emits a collection ID. It is
idempotent and copies no images. Register an existing incremental projection next,
if present, without calling `materializeProjection`:

```bash
python -m futsal registerMedia \
  --storeDir "$workRoot/data/mediaStore" \
  --sourceDir "$workRoot/data/projections/separatedV1"
```

The projection's parent must already be registered. If the supplement is not on
the receiver, use incremental transport below. An existing native plan directory
containing `plan.json` can also be registered. Missing reviewed development frames
can then be extracted once with `prepareView --dataRoot <registeredRawRoot>`.
Public image files are bound directly; existing original inputs remain intact.
Reserved test frames are never extracted by view preparation.

## Prepare reusable views

```bash
python -m futsal prepareView \
  --storeDir "$workRoot/data/mediaStore" \
  --config configs/separatedDetection.yaml \
  --outputDir "$workRoot/data/views/separatedV1"
python -m futsal verifyDataset \
  --datasetDir "$workRoot/data/views/separatedV1" \
  --mediaStoreDir "$workRoot/data/mediaStore"
```

`--outputDir` must be new. To select the native template explicitly, supply the
collection ID returned at registration using `--collectionId`. Otherwise the
original plan with the broadest source inventory is preferred. Repeating a recipe
over the same pinned template and assets produces the same view identity.

| Recipe | Active task views | Training sources |
| --- | --- | --- |
| `configs/curriculum.yaml` | genericObjects, roleObjects, unified25 | Existing per-stage public/own declarations |
| `configs/separatedDetection.yaml` | detailedObjects, courtPoints | Own real data |
| `configs/sharedDetection.yaml` | detailedObjects, genericObjects, playerOnly, ballOnly, courtPoints | Own real data |

`sharedDetection` is a reusable collection of independent task views, not a new
scientific curriculum schedule. Select one task with its training profile. The
four detailed athlete classes remain distinct in detailedObjects; genericObjects
and playerOnly collapse them to player. ballOnly keeps only ball annotations.
All views derive eligibility from native annotations, preserving task-specific
quarantine, annotation completeness, frozen match splits and point box rules.

## Train and evaluate without runtime image copies

Use the existing official initial weights directory and a new run output:

```bash
python -m futsal train \
  --datasetDir "$workRoot/data/views/separatedV1" \
  --mediaStoreDir "$workRoot/data/mediaStore" \
  --config configs/training/courtPoints.yaml \
  --weights "$workRoot/data/modelWeights/yolo26n/yolo26n.pt" \
  --outputDir "$workRoot/runs/courtReal42"
```

For independent object tasks, prepare sharedDetection and choose
`configs/training/detailedObjects.yaml`, `genericObjects.yaml`, `playerOnly.yaml`
or `ballOnly.yaml`. These profiles use the existing independent training defaults;
they do not initialize from the completed curriculum. Seeds and model runs reuse
the same image files. The original curriculum recipe/profile remains available.

The automatic report evaluation receives the same media store. Explicit
`evaluateRun` and `evaluateConfiguration` calls also accept `--mediaStoreDir`.
Joint detailed-object/court evaluation can reuse an existing legacy curriculum
as `--referenceDatasetDir`; its images are read through registered asset hashes.
The generic/player/ball task profiles do not add new joint-comparison ontologies.

The native reader uses view targets and preserves native transforms, batching and
metrics. It carries record IDs through batches, so different logical samples
sharing bytes remain different samples. Source-adjacent caches are neither created
nor consumed, and incomplete JPEGs fail instead of being repaired. Keep cache=false.
Run metadata snapshots include the selected records, native plan, view contract,
configuration, source/code evidence and resolved media locations. Checkpoints and
bounded portable report galleries are still saved; full runtime image copies are not.

## Optional pre-generated synthetic court data

The generator remains independent: run `scripts/generateSyntheticCourt.py` as
described in [synthetic court data](syntheticCourt.md), then wait for completion.
Add the completed source to a new invocation of the same court training command:

```text
--syntheticDir /home/user_marco/futsalWork/data/generated/syntheticCourtV1
```

The shared workflow verifies and registers that completed collection in place.
Generation is never invoked by training. The manifest controls the image count
and seed, including a 2,500-image seed42 source. Partial outputs are rejected.
Only training gains synthetic records; the real court validation stays identical.
Each invocation uses fresh official initialization and optimizer state. Real-only
training can continue while the standalone generator is running. Registering a
completed collection does not invalidate another run's pinned input selection.

## Incremental transport and relocation

On the receiver, register existing datasets once and write a verified inventory:

```text
python scripts/sharedDataTransfer.py inventory --storeDir <receiverStore> --output <inventory.json>
```

Move that small inventory to the sender. Package a view and only its absent assets:

```text
python scripts/sharedDataTransfer.py pack --datasetDir <view> --storeDir <senderStore> --receiverInventory <inventory.json> --archive <view.tar.gz> --receipt <view.receipt.json>
python scripts/sharedDataTransfer.py unpack --storeDir <receiverStore> --archive <view.tar.gz> --receipt <view.receipt.json> --outputDir <newReceivedDirectory>
```

The received view is `<newReceivedDirectory>/view`. Missing images remain under
that received directory's `assets`; registration binds them in place. Keep that
directory as an immutable media provider. Receipt, archive hashes, paths, membership,
selected asset hashes and existing receiver assets are verified before use. A stale
inventory fails on receipt instead of creating a partial usable view. Transfer the
archive and receipt once; subsequent packages can consist of task metadata only.

Bindings are machine-local and excluded from scientific identities. After moving
an asset between runs, explicitly rebind verified bytes:

```text
python -m futsal rebindMedia --storeDir <store> --assetId <sha256> --root <newRoot> --relativePath <relativeImagePath>
```

Never move media used by an active run. A media-store writer lock rejects concurrent
updates; retry after the writer finishes. After an externally killed writer, confirm
no writer remains before removing its `.writeLock`; ordinary failures release it.

## Compatibility and storage limits

Shared views have schemaVersion2 and kind sharedTaskView. Schema1 prepared bundles
and their original copy-based training remain supported. Supplying --mediaStoreDir
also enables direct registered-media loading for an existing legacy bundle, while
retaining its contract identity. Its selected assets must already be registered.
Existing archived runs can
be evaluated through a shared view only after checking equivalent ordered selections,
image bytes, targets, task maps, source exposure and frozen split against their
archived native plan or original dataset. The new evaluation records this equivalence;
old run/evaluation contracts are never rewritten. Continuation still requires exact
code, configuration, selection, initialization and view identity.

Existing duplicates are preserved. This workflow avoids new copies; it does not
reclaim old prepared/runtime storage. Newly missing real frames are published once
in the store under content hashes. Source originals, the store, received asset
providers and generated collections must remain available for training/inference.
Model-free archived report rendering needs only its existing portable evidence.
