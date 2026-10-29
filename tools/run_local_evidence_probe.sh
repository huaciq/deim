#!/usr/bin/env bash
# One explicit background run, or resume its saved protocol. Activate deim_zxy first.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

if [[ "${1:-}" == --resume && $# -eq 2 ]]; then
    OUT=$(realpath -- "$2")
    LOG="$OUT/resume_$(date +%Y%m%d_%H%M%S).console.log"
    ARGS=(resume --output "$OUT")
elif [[ $# -ge 4 && $# -le 5 ]]; then
    CONFIG=$1
    CHECKPOINT=$2
    DATA_ROOT=$3
    ANN=$4
    EXP_ROOT=${EXPERIMENT_ROOT:-/home/zxy/sar/experiments}
    mkdir -p -- "$EXP_ROOT"
    OUT=$(mktemp -d "$EXP_ROOT/local_evidence_probe_$(date +%Y%m%d_%H%M%S)_XXXXXX")
    LOG="$OUT/console.log"
    ARGS=(run -c "$CONFIG" -r "$CHECKPOINT" --img-folder "$DATA_ROOT" --ann "$ANN" --output "$OUT")
    if [[ $# -eq 5 ]]; then ARGS+=(--groups "$5"); fi
else
    printf '%s\n' 'Usage: bash tools/run_local_evidence_probe.sh CONFIG CHECKPOINT IMAGE_ROOT VAL_JSON [SCENE_GROUPS_JSON]' \
                  '       bash tools/run_local_evidence_probe.sh --resume OUT' >&2
    exit 2
fi

nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 \
    torchrun --standalone --nproc_per_node=2 tools/local_evidence_probe.py "${ARGS[@]}" \
    > "$LOG" 2>&1 < /dev/null &
LAUNCHER_PID=$!
printf '%s\n' "$LAUNCHER_PID" > "$OUT/launcher.pid"
disown "$LAUNCHER_PID"
printf 'STARTED OUT=%s\nPID=%s\nLOG=%s\n' "$OUT" "$LAUNCHER_PID" "$LOG"
