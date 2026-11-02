#!/usr/bin/env bash
# Sequential two-GPU inference followed by CPU paired analysis; no training.
set -e
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy

if [ "$#" -eq 2 ] && [ "$1" = "--resume" ]; then
    OUT=$(realpath "$2")
    LOG="$OUT/resume_$(date +%Y%m%d_%H%M%S).console.log"
    ARGS=(resume --output "$OUT")
elif [ "$#" -eq 2 ]; then
    DATA_ROOT=${DATA_ROOT:-/home/zxy/sar/datasets/SAR-AIRcraft-1.0}
    DIAGNOSIS_ROOT=${DIAGNOSIS_ROOT:-/data2/zxy/sar/experiments/diagnostics}
    mkdir -p "$DIAGNOSIS_ROOT"
    SPLIT=${SPLIT:-val}
    OUT=$(mktemp -d "$DIAGNOSIS_ROOT/rem102_comparison_${SPLIT}_$(date +%Y%m%d_%H%M%S)_XXXXXX")
    LOG="$OUT/console.log"
    ARGS=(run --baseline-run "$1" --rem-run "$2" --data-root "$DATA_ROOT" --output "$OUT" --split "$SPLIT")
else
    printf '%s\n' 'Usage: [SPLIT=val|test] run_rem_102e_diagnosis.sh BASELINE_RUN REM_RUN | --resume OUT' >&2
    exit 2
fi
ulimit -Sn 65535
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 \
    python tools/compare_rem_102e.py "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
printf 'STARTED OUT=%s PID=%s LOG=%s\n' "$OUT" "$PID" "$LOG"
