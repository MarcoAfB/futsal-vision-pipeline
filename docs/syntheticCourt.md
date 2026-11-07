# Synthetic court generation and optional training supplementation

There is one twenty-class court detector, one training routine and one shared
profile. The optional `--syntheticDir` changes only training inputs. Generation
is a separate operation and never starts training.

## Generate and verify independently

Run inside the project with its installed environment:

```text
python scripts/generateSyntheticCourt.py --count 2000 --seed 42 --outputDir data/generated/syntheticCourtV1
python scripts/generateSyntheticCourt.py --verify --outputDir data/generated/syntheticCourtV1
```

Defaults in `configs/syntheticCourt.yaml` are 2000 images, seed 42, 1280 by 720,
texture scale 48 and sixteen annotated previews. Optional arguments include
`--config`, `--count`, `--seed`, `--width`, `--height`, `--textureScale` and
`--debugCount`. Only base project dependencies are needed; generation imports no
model backend and reads no real match media or annotations.

```text
syntheticCourtV1/
  images/train/syntheticCourt000000.jpg
  labels/train/syntheticCourt000000.txt
  debug/syntheticCourt000000.jpg
  configuration.json
  manifest.jsonl
  summary.json
  contract.json
```

Training images have no target markers/names. Annotated previews remain in debug/
and never enter training. The manifest records native projected points, camera,
photometry, visible references and per-image seeds. Detection labels use five
YOLO fields and the current real-data 1/64 by 1/36 point boxes. Native points are
preserved when derived boxes are clipped at image boundaries.

The physical model defaults to 40 by 20 metres, with fixed Side A/B and canonical
class order. Rendering varies floors, perspective, mild lens distortion, boards,
goals, people, shadows and broadcast effects. Camera priors are inherited
rendering defaults; they are not fitted to the current training split. This
provenance is recorded explicitly. Subsequent fitting must use training data only.

Per-image seeds and least-supported-reference targeting make generation
reproducible under the recorded software/source versions. At least twenty images
require all twenty classes to be covered. Publication is atomic after native
annotation, image, membership and hash verification. Existing output directories
are refused. Raw sources and existing prepared bundles stay unchanged.

## Two calls of the same training routine

Materialize the real separated dataset following [separated detection](separatedDetection.md).
Generate or receive the synthetic source separately, then run:

```text
python -m futsal train --datasetDir data/prepared/separatedV1 --config configs/training/courtPoints.yaml --weights data/modelWeights/yolo26n/yolo26n.pt --outputDir outputs/training/courtReal
python -m futsal train --datasetDir data/prepared/separatedV1 --config configs/training/courtPoints.yaml --weights data/modelWeights/yolo26n/yolo26n.pt --syntheticDir data/generated/syntheticCourtV1 --outputDir outputs/training/courtRealSynthetic
```

Both calls share the configuration and official YOLO26n initialization. The first
uses 2579 real training records; the second adds exactly 2000 generated records,
for 4579. Both retain the same 548 real validation records. Frozen match splits,
quarantine and the real raw-source contract remain unchanged.

The optional argument resolves the run policy to `realWithSyntheticCourt` and
archives the override. The original YAML remains `realOnly`. Supplementation is
restricted to independent scientific courtPoints training; other profiles retain
their existing restrictions. Missing, incompatible or corrupt synthetic sources
fail explicitly. No generation is triggered by a train command.

Inputs include the synthetic contract/manifest. Method and selection identities
bind their exact contents; source/runtime integrity is checked before and after
training. Each execution produces its own checkpoints with a fresh optimizer.
Shared defaults remain seed 42, size 960, batch 2/nbs 64, FP32 SGD, maximum 100
epochs, patience 20 and disabled augmentations. Best-checkpoint selection remains
native validation mAP50-95. The object detector is still trained once on real data.

Allow space for the materialized real bundle, generated source, selected runtime
copies, checkpoints, validation workspaces and the existing 2 GiB reserve.
Supplementation does not create another complete prepared real dataset. Follow
the locked environment and GPU checks in [server training](serverTraining.md).

## Sealed validation comparison

Training generates its real-validation report through the existing workflow.
Compare the two completed evaluations without further inference:

```text
python -m futsal compareCourtTraining --baselineEvaluationDir <realEvaluation> --candidateEvaluationDir <mixedEvaluation> --outputDir outputs/reports/courtComparison
python -m futsal reportRun --evaluationDir outputs/reports/courtComparison --outputDir outputs/reports/courtComparisonRegenerated
```

Comparison requires identical real training membership, full validation references,
official initialization, source code and hyperparameters. Only synthetic training
membership may differ. Tables include detection, native-point success/errors/misses
and results by original match. Unsupported lowerCornerSideB remains N/A. Epochs,
batches, actual optimizer updates and elapsed training time are included: matching
epoch limits does not equalize computation. One seed establishes no significance.

The existing evaluateConfiguration accepts either court run with the same real-only
objects run and real dataset, preserving the 539 common canonical validation records
and native composition/matching rules. Its evidence archives synthetic provenance.
The existing curriculum comparison keeps its original purpose and command.

## Independent data and result transport

The synthetic transfer tool uses only the standard library:

```text
python scripts/syntheticTransfer.py pack --datasetDir data/generated/syntheticCourtV1 --archive data/transfers/syntheticCourtV1.tar.gz --receipt data/transfers/syntheticCourtV1.receipt.json
python scripts/syntheticTransfer.py unpack --archive incoming/syntheticCourtV1.tar.gz --receipt incoming/syntheticCourtV1.receipt.json --outputDir data/generated/syntheticCourtV1
python scripts/generateSyntheticCourt.py --verify --outputDir data/generated/syntheticCourtV1
```

Transfer checks membership and hashes; final native verification also checks image
decoding and point/label consistency. Existing resultTransfer --runDir packages
include the mixed run's archived source contract/manifest. Attach its validation
and court comparison with repeated --reportDir. Joint --configurationDir packages
also retain the mixed component's source evidence. Runtime images remain excluded
from result packages; receivers do not claim verification of omitted bytes.
Reports regenerate without model packages or the original synthetic source folder.
