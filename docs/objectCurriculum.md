# Public-to-own object curriculum

`configs/objectCurriculum.yaml` defines a detector curriculum with ball in every
stage. Each selected YOLO26 size starts independently from its matching official
weights and transfers the previous stage's best checkpoint to the next stage.

| Stage | Classes | Training sources | Maximum epochs | lr0 | Patience |
| --- | --- | --- | --- | --- | --- |
| genericObjects | ball, player | Futsal4400, SmartFutsal | 80 | 0.001 | 15 |
| roleObjects | ball, outfieldPlayer, goalkeeper | FutsalMOT, FutsalAnaliza, FutsalDetect | 60 | 0.0005 | 15 |
| detailedObjects | ball, outfieldPlayerTeamA, outfieldPlayerTeamB, goalkeeperTeamA, goalkeeperTeamB | own training matches | 100 | 0.0002 | 20 |

All stages use own validation matches and native annotations. Team A/B is the
recording's annotation identity, not a fixed court side or a universal club ID.
`unified25` remains declared as inactive canonical coverage so the frozen split
and quarantine basis stay unchanged. No court detector is trained by this recipe.
The final five-class best checkpoint is the resulting detector; stage metrics
describe different ontologies and cannot establish a curriculum benefit.

The scientific profile uses seed42, 960px, FP32, SGD, nominal batch64,
deterministic execution and disabled augmentations. Default order is n/s/m/l/x.
These are independent models, with no transfer between sizes. Early stopping can
finish a stage before its maximum epoch count. This training does not train or
embed a tracker; tracking remains an inference-time workflow.

## Prepare a new shared view

From a checkout with the locked environment installed, reuse the existing native
curriculum plan and registered media providers. Run from the repository root:

```bash
PYTHON=.venv/bin/python
WORK=../futsalWork
"$PYTHON" scripts/runObjectCurriculum.py prepare \
  --sourceDatasetDir "$WORK/data/views/curriculumV1" \
  --mediaStoreDir "$WORK/data/mediaStore" \
  --planDir "$WORK/data/plans/objectCurriculumV1" \
  --datasetDir "$WORK/data/views/objectCurriculumV1"
```

Use the actual existing curriculum bundle or native-plan directory if its path
differs. Register existing curriculum and own-data supplements with
`python -m futsal registerMedia` first if they are not yet in the store. Preparation
requires all eligible image providers already bound locally; it writes a new
native plan, registers that exact collection, then writes new task labels and
metadata. It never copies original media or changes earlier contracts. Existing
output directories are refused. Keep the derived plan directory as a registered
provider. The generic `prepareProjection` path remains own-data-only.

For the current registered source snapshot, expected exported training counts are
3398 / 1989 / 2553 and validation counts are 541 / 541 / 541. The resulting view
has 9563 task rows referring to 8481 unique image hashes. These are audit values,
not constants in the implementation; the manifest and contract are authoritative.
The reserved test images and labels are never exported or evaluated.

## Run and resume

Use a clean committed checkout and existing official weight sidecars under
`$WORK/data/modelWeights/yolo26{size}/`. The launcher never downloads packages or
weights. Its isolated workers check the pinned environment and requested GPU
before training. CPU fixtures verify software behavior separately from science.

```bash
CAMPAIGN="$WORK/launch/objectCurriculumAllSizesSeed42_$(date -u +%Y%m%dT%H%M%SZ)"
"$PYTHON" scripts/runObjectCurriculum.py train \
  --workDir "$WORK" --datasetDir "$WORK/data/views/objectCurriculumV1" \
  --mediaStoreDir "$WORK/data/mediaStore" --campaignDir "$CAMPAIGN"
# Resume the same campaign with all inputs, code and weights unchanged:
"$PYTHON" scripts/runObjectCurriculum.py train \
  --workDir "$WORK" --datasetDir "$WORK/data/views/objectCurriculumV1" \
  --mediaStoreDir "$WORK/data/mediaStore" --campaignDir "$CAMPAIGN" --resume
```

Select fewer sizes with `--modelSizes x` or `--modelSizes n s`. Use `--weightsRoot`
for a different existing weight root. Windows uses `.venv/Scripts/python.exe`.
Runs are grouped under `$WORK/runs/<campaign>/yolo26{size}RealAttemptNN`; reports
are grouped under `$WORK/reports/<campaign>/<attempt>/`. Worker logs, frozen
profiles, identities and training/report outcomes are recorded in `campaignDir`.
Keep this parent distinct from earlier standalone detailedObjects campaigns.

Each stage starts at batch8. A definite OOM/allocation failure permits a fresh
attempt at batch4, then 2, then 1 for that failed stage only. The next stage starts
at 8. Completed stages are copied with verified artifact hashes and checkpoint
lineage; the failed stage restarts at epoch zero with a fresh optimizer, using
the preceding best checkpoint or official weights for stage1. Partial optimizer
state and partial failed checkpoints are never resumed. Batch1 exhaustion marks
that model failed and advances to the next size. Ambiguous CUDA errors never
authorize batch reduction. Every attempt, including failures, is preserved.

Core training defaults to `--batchRecovery native` for existing workflows. This
launcher always uses `fixed`, blocking native silent reductions and retries,
including a repeated batch1. Scientific profiles support stage-specific
`parameters.batch`. `--memoryRetryFrom` requires verified memory failure and
permits changing only the first failed stage's batch along 8/4/2/1; it is mutually
exclusive with the existing strict `--continueFrom`.

An interruption stops the queue. Explicit campaign resume can strictly reuse its
completed prefix at the same batch. A hard-killed, unsealed `running` attempt
requires inspection. `campaign.lock` prevents simultaneous launches; after a
hard kill, remove it only after verifying that all campaign workers have stopped.
Identity changes refuse resume. A failed report after completed training retries
postprocessing only; a sealed completed evaluation is rendered without inference.

All three stages receive class/support/source diagnostics and batch/lineage
tables. `reportRun` remains inference-free, and existing `resultTransfer.py`
supports these runs and their sealed reports with optional `--launchDir` evidence.
Campaign preflight and local tests are not evidence of GPU feasibility or model
quality. Actual scientific training, final test evaluation and tracking campaigns
are separate executions.
