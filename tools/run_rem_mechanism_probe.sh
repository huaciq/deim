#!/usr/bin/env bash
# Five sequential frozen-checkpoint exports; no training or parameter updates.
set -e
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
if [ "$#" -eq 2 ] && [ "$1" = "--resume" ]; then
    OUT=$(realpath "$2")
    LOG="$OUT/resume_$(date +%Y%m%d_%H%M%S).console.log"
    ARGS=(resume --output "$OUT")
elif [ "$#" -eq 1 ]; then
    DIAGNOSIS_ROOT=${DIAGNOSIS_ROOT:-/data2/zxy/sar/experiments/diagnostics}
    mkdir -p "$DIAGNOSIS_ROOT"
    OUT=$(mktemp -d "$DIAGNOSIS_ROOT/rem102_mechanism_$(date +%Y%m%d_%H%M%S)_XXXXXX")
    LOG="$OUT/console.log"
    ARGS=(run --comparison "$1" --output "$OUT")
else
    printf '%s\n' 'Usage: run_rem_mechanism_probe.sh COMPARISON_DIR | --resume OUT' >&2
    exit 2
fi
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
    python tools/rem_mechanism_probe.py "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
printf 'STARTED OUT=%s PID=%s LOG=%s\n' "$OUT" "$PID" "$LOG"
