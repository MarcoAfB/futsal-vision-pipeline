#!/usr/bin/env bash
# Embed torso crops of each condition of a sealed trackRun campaign and assign teams and goalkeepers
# per detection and per track. Requires FUTSAL_TRACKING_EVALUATION, FUTSAL_HOMOGRAPHY_EVALUATION,
# FUTSAL_REFERENCE_EVALUATION, the SigLIP snapshot in the local Hugging Face cache and the
# teamFeatures and teamClustering extras; see docs/teamRoles.md.
# FUTSAL_TRACK_CONDITIONS (space separated) narrows the conditions; by default every completed
# condition runs, player_botsort (the declared primary) first.
(
set -euo pipefail
cd "${FUTSAL_CODE:-/home/user_marco/futsal-vision-pipeline}"
export FUTSAL_WORK="${FUTSAL_WORK:-/home/user_marco/futsalWork}"
export UV_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
pythonBin="$PWD/.venv/bin/python"
[[ -x "$pythonBin" && -f configs/evaluation/teamFeatures.yaml && -f src/futsal/teams/trackRoles.py ]] || {
  echo 'Use a checkout with the track-level team code and its .venv.' >&2
  exit 2
}
for name in FUTSAL_TRACKING_EVALUATION FUTSAL_HOMOGRAPHY_EVALUATION FUTSAL_REFERENCE_EVALUATION; do
  [[ -n "${!name:-}" && -f "${!name}/evaluationContract.json" ]] || {
    echo "Set $name to a sealed evaluation directory." >&2
    exit 2
  }
done
commit="$(git rev-parse HEAD)"
checkCheckout() {
  [[ "$(git rev-parse HEAD)" == "$commit" && -z "$(git status --porcelain)" ]] || {
    echo 'The checkout must stay clean and at the same commit.' >&2
    exit 2
  }
}
checkCheckout

conditions="$("$pythonBin" - "$FUTSAL_TRACKING_EVALUATION" "${FUTSAL_TRACK_CONDITIONS:-}" <<'PY'
import json
import sys
from pathlib import Path

import futsal
import sklearn  # noqa: F401
import umap  # noqa: F401
from futsal.evaluation.teamFeatureRun import loadTeamFeatureConfig
from futsal.teams.appearanceFeatures import siglipFiles
from huggingface_hub import snapshot_download

if Path(futsal.__file__).resolve().parents[2] != Path.cwd().resolve():
    raise SystemExit('The .venv must import futsal from this checkout.')
config = loadTeamFeatureConfig('configs/evaluation/teamFeatures.yaml')[0]
try:
    snapshot_download(config['siglipModel'], revision=config['siglipRevision'], local_files_only=True,
                      allow_patterns=list(siglipFiles))
except Exception as error:
    raise SystemExit(f'SigLIP snapshot is not cached locally ({error}); see docs/teamRoles.md.') from error
state = json.loads((Path(sys.argv[1]) / 'evaluation.json').read_text(encoding='utf-8'))
completed = [item['conditionId'] for item in state['conditions'] if item['status'] == 'completed']
requested = sys.argv[2].split() or completed
missing = sorted(set(requested) - set(completed))
if missing:
    raise SystemExit(f'Conditions not completed in the campaign: {missing}; available: {completed}.')
print(' '.join(sorted(requested, key=lambda name: (not name.endswith('_botsort'), completed.index(name)))))
PY
)"

checkCheckout
outputRoot="${FUTSAL_TRACK_TEAMS_DIR:-$FUTSAL_WORK/reports/trackTeams/$(date -u +%Y%m%dT%H%M%S)_$$}"
printf 'Track-level team assignment: %s\nConditions: %s\n' "$outputRoot" "$conditions"
for condition in $conditions; do
  "$pythonBin" -m futsal extractTeamFeatures --trackingDir "$FUTSAL_TRACKING_EVALUATION" --condition "$condition" \
    --mediaStoreDir "$FUTSAL_WORK/data/mediaStore" --outputDir "$outputRoot/$condition/features" \
    --device "${FUTSAL_DEVICE:-cuda:0}"
  "$pythonBin" -m futsal evaluateTeams --featuresDir "$outputRoot/$condition/features" \
    --homographyEvaluationDir "$FUTSAL_HOMOGRAPHY_EVALUATION" \
    --referenceEvaluationDir "$FUTSAL_REFERENCE_EVALUATION" --outputDir "$outputRoot/$condition/evaluation"
  checkCheckout
done
)
