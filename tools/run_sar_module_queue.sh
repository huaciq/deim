#!/usr/bin/env bash
# Run several configs one after another on the same GPUs with run_sar_module_experiment.sh.
# For each config: wait until every GPU in GPU_IDS is idle, launch, then wait for that run's training process to exit.
# A launch refused by the GPU gate is retried after POLL_SECONDS. A failed run does not stop the queue.
# Usage: GPU_IDS=4,5 [EXPERIMENT_ROOT=.. INITIAL_WEIGHTS=.. RUN_PREFIX=.. POLL_SECONDS=300] bash tools/run_sar_module_queue.sh CONFIG...
set -u
GPU_IDS=${GPU_IDS:?set GPU_IDS, e.g. 4,5}
export GPU_IDS
POLL=${POLL_SECONDS:-300}
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

gpus_idle() {
    local apps uuid mem util
    apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
    for i in ${GPU_IDS//,/ }; do
        read -r uuid mem util < <(nvidia-smi -i "$i" --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ',')
        if [ "$mem" -gt 256 ] || [ "$util" -ne 0 ] || grep -q "$uuid" <<<"$apps"; then
            return 1
        fi
    done
}

stamp() { TZ=Asia/Shanghai date '+%F %T'; }

for config in "$@"; do
    echo "$(stamp) queued $config on GPUs $GPU_IDS"
    while true; do
        until gpus_idle; do sleep "$POLL"; done
        output=$(bash "$REPO_ROOT/tools/run_sar_module_experiment.sh" "$config" 2>&1)
        echo "$output"
        run=$(sed -n 's/^STARTED OUT=\([^ ]*\) PID=.*/\1/p' <<<"$output")
        [ -n "$run" ] && break
        echo "$(stamp) not started, retrying in ${POLL}s"
        sleep "$POLL"
    done
    pid=$(cat "$run/launcher.pid")
    echo "$(stamp) running $run (PID $pid)"
    while kill -0 "$pid" 2>/dev/null; do sleep "$POLL"; done
    echo "$(stamp) finished $run"
    sleep 120
done
echo "$(stamp) queue done"
