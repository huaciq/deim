#!/usr/bin/env bash
# Explicit, one-time launch. It never queues, waits for GPUs, or stops jobs.
# Usage: bash tools/run_sar_module_experiment.sh CONFIG
#        bash tools/run_sar_module_experiment.sh --resume RUN
set -e
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy

if [ "$#" -eq 1 ] && [ "$1" != "--resume" ]; then
    MODE=new
    CONFIG=$1
    INITIAL=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
    NAME=$(basename "$CONFIG")
    NAME=${NAME%.*}
    OUT=$(mktemp -d "/root/zxy/experiments/deim_m_sar_aircraft_${NAME}_seed0_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_XXXXXX")
    ATTEMPT=new
    LOG="$OUT/console.log"
    GATE="$OUT/gpu_gate.json"
    COMMAND="$OUT/start_command.sh"
elif [ "$#" -eq 2 ] && [ "$1" = "--resume" ]; then
    MODE=resume
    OUT=$(realpath "$2")
    CONFIG="$OUT/resolved_config.yml"
    INITIAL="$OUT/last.pth"
    ATTEMPT="resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
    LOG="$OUT/${ATTEMPT}.console.log"
    GATE="$OUT/${ATTEMPT}.gpu_gate.json"
    COMMAND="$OUT/${ATTEMPT}.command.sh"
else
    printf '%s\n' 'Usage: run_sar_module_experiment.sh CONFIG | --resume RUN' >&2
    exit 2
fi
export MODE CONFIG INITIAL OUT ATTEMPT LOG GATE COMMAND

python - <<'PY'
import json
import os
from pathlib import Path
import subprocess
import yaml

from engine.core.yaml_utils import load_config

run = Path(os.environ['OUT'])
source = Path(os.environ['CONFIG']).resolve()
mode = os.environ['MODE']
cfg = load_config(str(source), {})
cfg.pop('__include__', None)
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if mode == 'new':
    cfg.update(output_dir=str(run), summary_dir=str(run / 'summary'))
    (run / 'source_config.yml').write_text(source.read_text(), encoding='utf-8')
    (run / 'config_path.txt').write_text(str(source) + '\n', encoding='utf-8')
    (run / 'resolved_config.yml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    (run / 'git_revision.txt').write_text(revision + '\n', encoding='utf-8')
    (run / 'git_worktree_status.txt').write_text(subprocess.check_output(['git', 'status', '--short'], text=True), encoding='utf-8')
    (run / 'method_and_protocol.md').write_text(Path('docs/sar_structure_frequency_experiments.md').read_text(encoding='utf-8'), encoding='utf-8')
    (run / 'dataset_stats.json').write_text(Path('/root/zxy/datasets/SAR-AIRcraft-1.0/COCO/stats.json').read_text(), encoding='utf-8')
elif not Path(os.environ['INITIAL']).is_file():
    raise FileNotFoundError('Resume requires the stopped run\'s last.pth: ' + os.environ['INITIAL'])

train = cfg['train_dataloader']
collate = train.get('collate_fn', {})
criterion = cfg.get('DEIMCriterion', {})
modules = {
    'scs': bool(cfg.get('DFINETransformer', {}).get('use_scs', False)),
    'slqc': bool(cfg.get('DFINETransformer', {}).get('use_slqc', False)),
    'slqc_mode': cfg.get('DFINETransformer', {}).get('slqc_mode', 'layout'),
    'dsg': bool(cfg.get('HybridEncoder', {}).get('use_dsg', False)),
    'dsg_mode': cfg.get('HybridEncoder', {}).get('dsg_mode', 'directional'),
}
metadata = dict(
    mode=mode, source_config=str(source), output_dir=str(run),
    current_git_revision=revision, baseline_revision='a448005', modules=modules,
    seed=cfg.get('seed', 0), devices=[0, 1], world_size=2,
    global_batch=train['total_batch_size'], batch_per_gpu=train['total_batch_size'] // 2,
    epochs=cfg['epoches'], evaluation=dict(split='val', spatial_size=cfg['eval_spatial_size'],
                                         global_batch=cfg['val_dataloader']['total_batch_size'],
                                         postprocessor=cfg.get('PostProcessor', {})),
    optimizer=cfg['optimizer'], lr_schedule=dict(name=cfg.get('lrsheduler'),
        warmup_iter=cfg.get('warmup_iter'), flat_epoch=cfg.get('flat_epoch'),
        no_aug_epoch=cfg.get('no_aug_epoch'), lr_gamma=cfg.get('lr_gamma')),
    augmentation=dict(transforms=train['dataset']['transforms'], collate=collate),
    ema=dict(enabled=cfg.get('use_ema'), settings=cfg.get('ema', {})),
    amp=dict(enabled=cfg.get('use_amp'), dtype=cfg.get('amp_dtype'),
             fp32_loss=cfg.get('amp_loss_fp32'), fp32_retry=cfg.get('amp_fp32_retry'), scaler=cfg.get('scaler', {})),
    initialization=os.environ['INITIAL'], initialization_mode='COCO tuning -t; fresh optimizer' if mode == 'new' else 'restore -r from saved configuration and last.pth',
    training_only_loss_weights={k:v for k,v in criterion.get('weight_dict', {}).items() if k.startswith('loss_scs')},
    resolved_config=cfg,
)
filename = 'run_metadata.json' if mode == 'new' else os.environ['ATTEMPT'] + '.metadata.json'
(run / filename).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(dict(output=str(run), mode=mode, modules=modules), ensure_ascii=False), flush=True)
PY

# The gate is checked once immediately before nohup. Other GPUs are untouched.
python - <<'PY'
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

def query(arguments):
    return list(csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi', *arguments], text=True))))

cards = query(['--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'])
apps = query(['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])
active = {row[0].strip() for row in apps if row}
state = [dict(index=int(i), uuid=u.strip(), memory_mib=int(m.strip()), utilization=int(v.strip()))
         for i, u, m, v in cards if int(i) in (0, 1)]
idle = len(state) == 2 and all(s['uuid'] not in active and s['memory_mib'] <= 256 and s['utilization'] == 0 for s in state)
Path(os.environ['GATE']).write_text(json.dumps(dict(both_idle=idle, gpus=state, compute_apps=apps), indent=2), encoding='utf-8')
print(json.dumps(dict(output=os.environ['OUT'], both_idle=idle, gpus=state)), flush=True)
if not idle:
    marker = 'NOT_STARTED.txt' if os.environ['MODE'] == 'new' else os.environ['ATTEMPT'] + '.NOT_STARTED.txt'
    (Path(os.environ['OUT']) / marker).write_text('GPU 0 and GPU 1 were not simultaneously idle. No process launched.\n', encoding='utf-8')
    sys.exit(3)
PY

python - <<'PY'
import os
from pathlib import Path
import shlex
import yaml

cfg = yaml.safe_load((Path(os.environ['OUT']) / 'resolved_config.yml').read_text())
arguments = [
    'torchrun', '--standalone', '--nproc_per_node=2', 'train.py',
    '-c', str(Path(os.environ['OUT']) / 'resolved_config.yml'),
    '-t' if os.environ['MODE'] == 'new' else '-r', os.environ['INITIAL'],
    '--seed', str(cfg.get('seed', 0)), '--use-amp',
    '--output-dir', os.environ['OUT'], '--summary-dir', str(Path(os.environ['OUT']) / 'summary'),
]
script = '\n'.join([
    '#!/usr/bin/env bash', 'set -e', 'cd /root/zxy/deim',
    'source /opt/miniconda3/etc/profile.d/conda.sh', 'conda activate deim_zxy',
    'ulimit -Sn 65535',
    'exec env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 ' + shlex.join(arguments), '',
])
Path(os.environ['COMMAND']).write_text(script, encoding='utf-8')
PY
ulimit -Sn 65535
ulimit -Sn > "$OUT/nofile_soft.txt"
nohup bash "$COMMAND" > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
if [ "$MODE" = "resume" ]; then
    printf '%s\n' "$PID" > "$OUT/${ATTEMPT}.pid"
fi
disown "$PID"
printf 'STARTED OUT=%s PID=%s LOG=%s\n' "$OUT" "$PID" "$LOG"
