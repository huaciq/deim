#!/usr/bin/env bash
# A one-time conditional launch, not a queue or a background GPU watcher.
set -e
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
CONFIG=configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_scs_m1_102e_b28.yml
INITIAL=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
OUT=$(mktemp -d /root/zxy/experiments/deim_m_sar_aircraft_scs_m1_102e_b28_seed0_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_XXXXXX)
export CONFIG INITIAL OUT
git rev-parse HEAD > "$OUT/git_revision.txt"
git status --short > "$OUT/git_worktree_status.txt"
cp "$CONFIG" "$OUT/source_config.yml"
cp docs/scs_m1.md "$OUT/method_and_protocol.md"
cp /root/zxy/datasets/SAR-AIRcraft-1.0/COCO/stats.json "$OUT/dataset_stats.json"
python - <<'PY'
import json, os, pathlib, yaml
from engine.core.yaml_utils import load_config
p = pathlib.Path(os.environ['OUT'])
c = load_config(os.environ['CONFIG'], {})
c.pop('__include__', None)
c.update(output_dir=str(p), summary_dir=str(p / 'summary'), seed=0)
(p / 'resolved_config.yml').write_text(yaml.safe_dump(c, sort_keys=False))
metadata = dict(method='M1/SCS', git_revision=(p / 'git_revision.txt').read_text().strip(),
                baseline_revision='a448005', seed=0, devices=[0, 1], world_size=2,
                global_batch=28, batch_per_gpu=14, epochs=102,
                input_base_and_eval=640, train_scales='480..800, step32, until epoch90',
                initialization=os.environ['INITIAL'], initialization_mode='COCO tuning -t, fresh optimizer',
                descriptor='28D, post-transform pre-MixUp/multiscale',
                pair_membership_source='fixed baseline VAL associations; no TEST predictions',
                pair_weight_source='TRAIN counts, row-normalized inverse sqrt rival frequency',
                train_class_counts={'A220':3023, 'A320/321':1637, 'A330':251, 'ARJ21':447,
                                    'Boeing737':1755, 'Boeing787':1930, 'other':2508},
                resolved_config=c)
(p / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
PY

# Check both requested GPUs immediately before launching. Busy => no start.
python - <<'PY'
import csv, io, json, os, pathlib, subprocess, sys
def query(arguments):
    return list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi', *arguments], text=True))))
cards = query(['--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
apps = query(['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])
active = {row[0].strip() for row in apps if row}
state = [dict(index=int(i), uuid=u.strip(), memory_mib=int(m.strip()), utilization=int(v.strip()))
         for i,u,m,v in cards if int(i) in (0,1)]
idle = len(state) == 2 and all(s['uuid'] not in active and s['memory_mib'] <= 256 and s['utilization'] == 0 for s in state)
p = pathlib.Path(os.environ['OUT'])
(p / 'gpu_gate.json').write_text(json.dumps(dict(both_idle=idle, gpus=state, compute_apps=apps), indent=2))
print(json.dumps(dict(output=str(p), both_idle=idle, gpus=state)), flush=True)
if not idle:
    (p / 'NOT_STARTED.txt').write_text('GPU 0 and GPU 1 were not simultaneously idle. No training launched.\n')
    sys.exit(3)
PY
ulimit -Sn 65535
ulimit -Sn > "$OUT/nofile_soft.txt"
cat > "$OUT/start_command.sh" <<EOF
#!/usr/bin/env bash
set -e
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
ulimit -Sn 65535
exec env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 train.py -c '$OUT/resolved_config.yml' -t '$INITIAL' --seed 0 --use-amp --output-dir '$OUT' --summary-dir '$OUT/summary'
EOF
nohup bash "$OUT/start_command.sh" > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
printf 'STARTED OUT=%s PID=%s\n' "$OUT" "$PID"
