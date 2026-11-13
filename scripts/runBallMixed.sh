#!/usr/bin/env bash
(
set -euo pipefail
cd "${FUTSAL_CODE:-/home/user_marco/futsal-vision-pipeline}"
export FUTSAL_WORK="${FUTSAL_WORK:-/home/user_marco/futsalWork}"
export FUTSAL_BALL_VIEW="${FUTSAL_BALL_VIEW:-$FUTSAL_WORK/data/views/ballMixedV1}"
export YOLO_OFFLINE=true YOLO_AUTOINSTALL=false UV_OFFLINE=1
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
pythonBin="$PWD/.venv/bin/python"
[[ -x "$pythonBin" && -f configs/training/ballMixed.yaml ]] || {
  echo 'Use the updated mixed-ball checkout and its .venv.' >&2
  exit 2
}
commit="$(git rev-parse HEAD)"
checkCheckout() {
  [[ "$(git rev-parse HEAD)" == "$commit" && -z "$(git status --porcelain)" ]] || {
    echo 'The checkout must stay clean and at the same commit.' >&2
    exit 2
  }
}
checkCheckout

"$pythonBin" - <<'PY'
import os
from pathlib import Path

import futsal
from futsal.datasets.ballMixed import buildBallMixedPlan
from futsal.datasets.datasetPlan import loadDatasetPlan
from futsal.datasets.mediaRegistration import registerMedia
from futsal.datasets.objectCurriculum import prepareObjectCurriculum
from futsal.datasets.recipes import loadRecipe
from futsal.datasets.sharedViews import prepareView, verifyView
from futsal.training.modelWeights import verifyWeights

if Path(futsal.__file__).resolve().parents[2] != Path.cwd().resolve():
    raise SystemExit('The .venv must import futsal from this checkout.')
work = Path(os.environ['FUTSAL_WORK']).resolve()
store = work / 'data/mediaStore'
view = Path(os.environ['FUTSAL_BALL_VIEW']).resolve()
planDir = work / 'data/plans/ballMixedV1'
recipePath = Path('configs/ballMixed.yaml').resolve()
recipe = loadRecipe(recipePath)
for size in ('n', 's', 'm', 'l', 'x'):
    verifyWeights(work / f'data/modelWeights/yolo26{size}/yolo26{size}.pt', size)
if view.exists():
    verifyView(view, store)
    if loadDatasetPlan(view / 'plan.json')['recipe'] != recipe.data:
        raise SystemExit('Existing ball view uses a different recipe; select a separate mixed view.')
    print(f'Reusing verified mixed ball view: {view}', flush=True)
else:
    source = Path(os.environ.get('FUTSAL_CURRICULUM_DATASET', work / 'data/prepared/curriculumV1')).resolve()
    if not (source / 'plan.json').is_file():
        raise SystemExit('Set FUTSAL_CURRICULUM_DATASET to the existing native public/own curriculum directory.')
    raw = Path(os.environ.get('FUTSAL_RAW_DATA_ROOT', work / 'data/raw')).resolve()
    dataRoot = raw if raw.is_dir() else None
    registerMedia(store, source)
    supplement = work / 'data/projections/separatedV1'
    if (supplement / 'projection.json').is_file():
        registerMedia(store, supplement)
    allTasks = work / 'data/views/allTasksV1'
    if (allTasks / 'contract.json').is_file():
        registerMedia(store, allTasks)
    try:
        if planDir.exists():
            expected = buildBallMixedPlan(loadDatasetPlan(source / 'plan.json'), recipe)
            if loadDatasetPlan(planDir / 'plan.json')['planId'] != expected['planId']:
                raise ValueError('Existing mixed ball plan differs from the current source/recipe.')
            registered = registerMedia(store, planDir)
            prepareView(store, recipePath, view, collectionId=registered['collectionId'], dataRoot=dataRoot)
        else:
            prepareObjectCurriculum(source, recipePath, store, planDir, view,
                                    curriculum='ballMixed', dataRoot=dataRoot)
    except (OSError, ValueError) as error:
        raise SystemExit(f'Mixed ball view preparation failed: {error}\n'
            'Supply existing raw inputs through FUTSAL_RAW_DATA_ROOT, or import a verified view '
            'with sharedDataTransfer.py and set FUTSAL_BALL_VIEW to its received view directory.') from error
    print(f'Prepared shared mixed ball view without copying registered images: {view}', flush=True)
PY

checkCheckout
campaignDir="${FUTSAL_CAMPAIGN_DIR:-$FUTSAL_WORK/launch/ballMixedAllSizesSeed42_$(date -u +%Y%m%dT%H%M%S)_$$}"
arguments=(train --workDir "$FUTSAL_WORK" --datasetDir "$FUTSAL_BALL_VIEW"
  --mediaStoreDir "$FUTSAL_WORK/data/mediaStore" --campaignDir "$campaignDir" --modelSizes n s m l x)
[[ ! -f "$campaignDir/campaign.json" ]] || arguments+=(--resume)
printf 'Campaign: %s\nReports: %s/reports/%s\n' "$campaignDir" "$FUTSAL_WORK" "$(basename "$campaignDir")"
printf 'Follow the announced worker log in another terminal: tail -n 80 -F <log>\n'
"$pythonBin" scripts/runBallTraining.py "${arguments[@]}"
)
