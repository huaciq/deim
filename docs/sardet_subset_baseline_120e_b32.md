# SARDet 子集 baseline：120 轮、双卡、batch 32

用户指定 GPU 0、1，全局 batch=32（每卡 16），训练 120 轮，从 COCO 检查点初始化。
使用固定六类子集：9,449 张训练图、4,096 张验证图，不更换划分。

配置：`configs/deim_dfine/deim_hgnetv2_s_sardet10pct_baseline_120e_b32.yml`。
基础输入 640，保留现有训练多尺度；AdamW 主学习率 0.0004、骨干 0.0002，
weight_decay=0.0001，lr_gamma=0.5。warmup=1000 iterations，flat=60 epochs，
最后 8 轮无强增强，切换点为 epoch 112。增强 policy=[4,60,112]，MixUp=[4,60]；
保留 Mosaic、PhotometricDistort、普通水平翻转及现有多尺度，移除 ZoomOut/IoUCrop。
EMA=True，AMP=True，seed=0；评估为原始 COCO bbox，不加平移校准。

这是新的 120 轮 batch-32 协议，原 40/80 轮 batch-28 配置保留。

## 更新代码和启动

服务器工作区已检查；保留原有无关未跟踪配置。遇到已跟踪修改或分叉先查看具体
内容，不执行 reset 或强制 checkout。已生成的子集直接复用，无需再次准备数据。

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
CONFIG=configs/deim_dfine/deim_hgnetv2_s_sardet10pct_baseline_120e_b32.yml
OUT=$(mktemp -d /root/zxy/experiments/deim_s_sardet10pct_baseline120e_b32_seed0_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_XXXXXX)
export CONFIG OUT
git rev-parse HEAD > "$OUT/git_revision.txt"
cp "$CONFIG" "$OUT/source_config.yml"
cp /root/zxy/experiments/data_subsets/sardet100k_screen_v1/stats.json "$OUT/subset_stats.json"
printf '0\n' > "$OUT/seed.txt"
python - <<'PY'
import os, pathlib, yaml
from engine.core.yaml_utils import load_config
p=pathlib.Path(os.environ['OUT'])
cfg=load_config(os.environ['CONFIG'],{})
cfg.pop('__include__',None)
cfg.update(output_dir=str(p),summary_dir=str(p/'summary'),seed=0,use_amp=True)
(p/'resolved_config.yml').write_text(yaml.safe_dump(cfg,sort_keys=False))
(p/'config_path.txt').write_text(str(p/'resolved_config.yml')+'\n')
PY
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" \
  -t /root/zxy/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --seed 0 --use-amp --output-dir "$OUT" \
  > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
printf 'OUT=%s\nPID=%s\n' "$OUT" "$PID"
```

## 重连、日志和指标

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换成启动命令输出的实际目录
export OUT
tail -f "$OUT/console.log"
```

另一个终端：

```bash
OUT=/root/zxy/experiments/替换成实际目录
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
watch -n 2 nvidia-smi
```

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换成实际目录
export OUT
python - <<'PY'
import json, os, pathlib
p=pathlib.Path(os.environ['OUT'])
rows=[json.loads(s) for s in (p/'log.txt').read_text().splitlines() if s.strip()]
for name,r in [('best',max(rows,key=lambda r:r['test_coco_eval_bbox'][0])),('latest',rows[-1])]:
    print(name,'epoch',r['epoch'],dict(zip(['AP','AP50','AP75','APs','APm','APl'],
          [round(100*x,3) for x in r['test_coco_eval_bbox'][:6]])))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

首次 epoch 评估完成后才有 log.txt 中的 AP。指标原值为 0..1，上述命令乘 100。
训练 COCO 评估数据写入 eval/latest.pth，对应最近评估，并不等同于最佳权重。

## 恢复训练

恢复到同一目录，使用固定的 resolved_config.yml 和 -r，不同时使用 -t。
当前 solver 的 last.pth 只在 epoch 112 前更新，正常新训练最后更新到 epoch 111；
它不代表最终轮。最后阶段中断时，可能需要从该保存点重复部分轮次。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换成实际目录
RESUME_LOG="$OUT/resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).console.log"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/last.pth" \
  --seed 0 --use-amp --output-dir "$OUT" \
  > "$RESUME_LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
tail -f "$RESUME_LOG"
```

## 检查点评估

分别评估实际存在的 best_stg1.pth / best_stg2.pth，以实测 AP 选择权重。
下面给出 stage1 的完整验证命令；stage2 存在时将 STAGE 改为 best_stg2，再执行同一块。
每次均创建新的评估目录，不覆盖训练产物。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换成实际目录
STAGE=best_stg1
EVAL=$(mktemp -d "$OUT/fullval_${STAGE}_XXXXXX")
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/$STAGE.pth" --test-only --use-amp \
  --output-dir "$EVAL" --summary-dir "$EVAL/summary" \
  -u val_dataloader.dataset.ann_file=/root/zxy/TCRA-SAR/datasets/SARDet_100K/Annotations/val.json \
  > "$EVAL/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$EVAL/launcher.pid"
disown "$PID"
tail -f "$EVAL/console.log"
```

该命令使用官方完整 10,492 张验证图，训练期间仍使用固定 4,096 张筛选验证图。
六类 AP 的读取方法见 [筛选协议交接](sardet_screening_handoff.md)。
