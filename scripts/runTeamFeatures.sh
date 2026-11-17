#!/usr/bin/env bash
# Embed torso crops of a sealed generic-player validation evaluation on the server.
# Requires FUTSAL_PLAYER_EVALUATION (a runValidation directory with a playerOnly stage)
# and the SigLIP snapshot in the local Hugging Face cache; see docs/teamRoles.md.
(
set -euo pipefail
cd "${FUTSAL_CODE:-/home/user_marco/futsal-vision-pipeline}"
export FUTSAL_WORK="${FUTSAL_WORK:-/home/user_marco/futsalWork}"
export UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
pythonBin="$PWD/.venv/bin/python"
[[ -x "$pythonBin" && -f configs/evaluation/teamFeatures.yaml ]] || {
  echo 'Use a checkout with the team feature code and its .venv.' >&2
  exit 2
}
[[ -n "${FUTSAL_PLAYER_EVALUATION:-}" && -f "$FUTSAL_PLAYER_EVALUATION/evaluationContract.json" ]] || {
  echo 'Set FUTSAL_PLAYER_EVALUATION to the sealed playerOnly runValidation directory.' >&2
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
from pathlib import Path

import futsal
from futsal.evaluation.teamFeatureRun import loadTeamFeatureConfig
from huggingface_hub import snapshot_download
from futsal.teams.appearanceFeatures import siglipFiles

if Path(futsal.__file__).resolve().parents[2] != Path.cwd().resolve():
    raise SystemExit('The .venv must import futsal from this checkout.')
config = loadTeamFeatureConfig('configs/evaluation/teamFeatures.yaml')[0]
try:
    snapshot_download(config['siglipModel'], revision=config['siglipRevision'], local_files_only=True,
                      allow_patterns=list(siglipFiles))
except Exception as error:
    raise SystemExit(f'SigLIP snapshot is not cached locally ({error}); download it once with network '
                     'access as described in docs/teamRoles.md.') from error
PY

checkCheckout
outputDir="${FUTSAL_TEAM_FEATURES_DIR:-$FUTSAL_WORK/reports/teamFeatures/playerM_$(date -u +%Y%m%dT%H%M%S)_$$}"
printf 'Team features: %s\n' "$outputDir"
"$pythonBin" -m futsal extractTeamFeatures --playerEvaluationDir "$FUTSAL_PLAYER_EVALUATION" \
  --mediaStoreDir "$FUTSAL_WORK/data/mediaStore" --outputDir "$outputDir" --device "${FUTSAL_DEVICE:-cuda:0}"
checkCheckout
)
