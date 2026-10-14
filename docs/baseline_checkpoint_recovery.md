# OGSOD baseline checkpoint recovery

The epoch-72 transition reads `best_stg1.pth` on every rank. Previously,
rank 1 could read it while rank 0 was still writing it. The fix synchronizes
ranks before resume reads and publishes checkpoint files by atomic replacement.
It changes no model, data, loss, or training hyperparameters.

## Transfer and server update

Source is transferred through origin/main. On the server:

```bash
cd /home/zxy/sar/repos/deim
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
conda activate deim_zxy
```

If the worktree has local changes or pull reports divergence, inspect those
changes before proceeding; do not reset or force checkout.

## Recover the interrupted run

The old run is preserved. The new directory must contain `best_stg1.pth` because
the solver also reads it during the no-augmentation stage. Use the original
run's seed if it differed from 0. No annotation preparation is needed.

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OLD=/data2/zxy/sar/experiments/deim_dfine_s_ogsod_baseline/seed0
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_baseline/seed0_resume_$(date +%Y%m%d_%H%M%S)
mkdir "$OUT"
cp -n "$OLD/best_stg1.pth" "$OUT/best_stg1.pth"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod.yml "$OUT/config.yml"
printf '%s\n' "resume=$OLD/last.pth" 'seed=0; global_batch=32; eval_batch=16; input=640 (training multiscale)' 'epochs=80; AdamW lr=0.0004; backbone lr=0.0002; flat_epoch=40; no_aug_epoch=8; warmup_iter=1000' 'Mosaic/photometric/zoom/IoU crop policy=[4,40,72]; mixup=[4,40]; stop_epoch=72' 'EMA=true decay=0.9999; AMP=true; initialization=COCO deim_dfine_hgnetv2_s_coco_120e.pth' 'evaluation=original OGSOD val.json, COCO bbox AP, EMA when available' > "$OUT/run_notes.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod.yml --seed 0 --use-amp \
  -r "$OLD/last.pth" --output-dir "$OUT" \
  > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
disown
echo "$OUT"
```

The pasted log shows rank 0 successfully loaded best_stg1, so an incomplete
read is more likely than permanent corruption. If the copied best_stg1 still
fails to load after all writers have exited, do not replace it with another
checkpoint and call that the same baseline protocol. Recover a valid copy of
the original best checkpoint, or document and rerun the affected experiment.

`last.pth` is saved only before epoch 72. If it is unavailable, an earlier
`checkpoint0069.pth` can restore an earlier epoch, with best_stg1 still required.

## Fresh baseline (alternative to recovery)

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_baseline/seed0_fresh_$(date +%Y%m%d_%H%M%S)
mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod.yml "$OUT/config.yml"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
disown
echo "$OUT"
```

Record the same run settings listed above, with COCO initialization in place
of resume. Never supply `-t` and `-r` together.

## Monitoring and evaluation

On reconnect explicitly activate the environment and set OUT to the directory
printed by the launcher:

```bash
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_baseline/REPLACE_WITH_PRINTED_RUN_DIRECTORY
tail -f "$OUT/console.log"
```

In another terminal:

```bash
watch -n 2 nvidia-smi
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 5 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 127.0.0.1 --port 6006
```

Print the best logged AP and its epoch (zero-based):

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

The resumed log contains only newly executed epochs; consult OLD/log.txt for
the earlier history. Epoch metrics are JSONL in log.txt, TensorBoard events
are in summary/, and evaluation details are in eval/latest.pth. Neither
last.pth nor eval/latest.pth is necessarily the best final model checkpoint.

Evaluate each available best checkpoint and select by evaluated AP, not its
stage filename. Each evaluation uses a fresh directory:

```bash
for CKPT in "$OUT/best_stg1.pth" "$OUT/best_stg2.pth"; do
  [ -f "$CKPT" ] || continue
  EVAL="$OUT/evaluation_$(basename "$CKPT" .pth)_$(date +%Y%m%d_%H%M%S)"
  mkdir "$EVAL"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
    -c /home/zxy/sar/repos/deim/configs/deim_dfine/deim_hgnetv2_s_ogsod.yml \
    -r "$CKPT" --test-only --output-dir "$EVAL" 2>&1 | tee "$EVAL/console.log"
done
```

Targeted local regression command:
`conda run -n pytorch python -X utf8=0 -m unittest discover -s tests -p test_checkpoint_io.py -v`.
Tests cover concurrent-reader visibility, interrupted writes, non-master saves,
and synchronization order. They do not establish successful two-GPU training
or availability of the server checkpoints.
