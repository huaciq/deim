# OGSOD 120 epoch baseline

Config: `configs/deim_dfine/deim_hgnetv2_s_ogsod_120e.yml`.
It inherits the 80 epoch baseline and overrides only the output path and schedule:
epoches=120, flat_epoch=60, augmentation policy=[4,60,112],
mixup_epochs=[4,60], stop_epoch=112. no_aug_epoch stays 8, warmup_iter stays 1000.
Epoch boundaries in the implementation are zero-based; epochs 112..119 are
the final eight epochs with the configured strong augmentation and multiscale
disabled. The first four epochs retain the original augmentation policy.

Other settings remain: seed=0 (launcher), global train batch=32 across two GPUs,
validation batch=16, input=640 with training multiscale before epoch 112,
AdamW lr=0.0004, backbone parameter group lr=0.0002, lr_gamma=0.5,
EMA enabled (decay=0.9999), AMP enabled, OGSOD original annotations,
COCO bbox evaluation using EMA when available. Initialization is
`/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth`.
Start this protocol from COCO initialization. Continuing an 80 epoch run under
this config has a different historical learning-rate and augmentation schedule.

## Server update and fresh training

```bash
cd /home/zxy/sar/repos/deim
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
conda activate deim_zxy
```

If the worktree has local changes or is diverged, inspect it before updating;
never use forced checkout or destructive reset. Original annotations already
work, so no preparation step is needed.

```bash
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_baseline_120e/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
python - "$OUT" <<'PY'
import pathlib, sys, yaml
from engine.core.yaml_utils import load_config
cfg = load_config('configs/deim_dfine/deim_hgnetv2_s_ogsod_120e.yml', {})
cfg.update(seed=0, output_dir=sys.argv[1], tuning='/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth')
cfg.pop('__include__', None)
with (pathlib.Path(sys.argv[1]) / 'resolved_config.yml').open('w') as stream:
    yaml.safe_dump(cfg, stream, sort_keys=False)
PY
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
disown
echo "$OUT"
```

## Resume an interrupted 120 epoch run

On reconnect activate the environment and set OLD to the actual 120 epoch
directory printed above. Use a fresh output directory and retain the stage-one
best checkpoint, because the solver reads it at and after epoch 112.

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OLD=/home/zxy/sar/experiments/deim_dfine_s_ogsod_baseline_120e/REPLACE_WITH_RUN_DIRECTORY
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_baseline_120e/seed0_resume_$(date +%Y%m%d_%H%M%S)
mkdir "$OUT"
cp -n "$OLD/best_stg1.pth" "$OUT/best_stg1.pth"
cp -n "$OLD/resolved_config.yml" "$OUT/original_resolved_config.yml"
git rev-parse HEAD > "$OUT/git_revision.txt"
printf '%s\n' "resume=$OLD/last.pth" "output_dir=$OUT" > "$OUT/resume_notes.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e.yml --seed 0 --use-amp \
  -r "$OLD/last.pth" --output-dir "$OUT" \
  > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
disown
echo "$OUT"
```

Use `-r` for resume, never also `-t`. `last.pth` updates only before epoch 112;
it is not a final-epoch checkpoint. If interrupted during the final stage,
this command repeats the final stage from that saved recovery point.

## Logs, metrics, processes, TensorBoard and evaluation

On reconnect explicitly set OUT and activate the environment:

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_baseline_120e/REPLACE_WITH_RUN_DIRECTORY
tail -f "$OUT/console.log"
```

Run each monitor in its own terminal with OUT set:

```bash
watch -n 2 nvidia-smi
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 5 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 127.0.0.1 --port 6006
```

```bash
python - "$OUT/log.txt" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
row = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
ap, ap50, ap75 = row['test_coco_eval_bbox'][:3]
print(f"epoch={row['epoch']} AP={100*ap:.3f} AP50={100*ap50:.3f} AP75={100*ap75:.3f}")
PY
```

Metrics are JSONL in log.txt; TensorBoard events are in summary/; COCO evaluation
details are in eval/latest.pth. A fresh resumed directory logs only the resumed
epochs, so retain and consult the original logs as well. Evaluate both available
best checkpoints and select by evaluated AP, not stage filename:

```bash
for CKPT in "$OUT/best_stg1.pth" "$OUT/best_stg2.pth"; do
  [ -f "$CKPT" ] || continue
  EVAL="$OUT/evaluation_$(basename "$CKPT" .pth)_$(date +%Y%m%d_%H%M%S)"
  mkdir "$EVAL"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
    -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e.yml \
    -r "$CKPT" --test-only --output-dir "$EVAL" 2>&1 | tee "$EVAL/console.log"
done
```

Validation scope: load and compare fully merged 80/120 epoch configs and check
schedule alignment. No local or remote full training is claimed.
