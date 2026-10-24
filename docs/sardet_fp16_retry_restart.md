# SARDet 子集 FP16 异常修复及从 COCO 重新启动

原训练目录：
`/root/zxy/experiments/deim_s_sardet10pct_baseline120e_b32_seed0_20261024_220731_eWczfP`。
原运行完成 epoch 0..5，epoch 5 的筛选验证 AP 为 27.7087%，epoch 6 中途退出。
用户要求重新开始，因此本次创建新实验目录、从 COCO 初始化，不读取原运行检查点。
原日志、检查点和指标保留。

## 修复范围

1. Mosaic 合并四张图的 image_id，结果是四元素张量。训练异常提示和 FP32 重算日志
   以前调用 .item()，会把原始非有限预测错误遮住。现改成按样本分组的 ID 列表，
   同时支持单 ID 和 Mosaic；普通验证的单图 image_id 行为不变。
2. 第一次错误来自非有限 pred_boxes。新增独立的
   `configs/deim_dfine/deim_hgnetv2_s_sardet10pct_baseline_120e_b32_fp16_retry.yml`，
   启用已有 amp_fp32_retry。两卡先统一判断；异常输出或 loss 只对同一批次重算一次
   FP32，恢复前向前的模型 buffer 和 torch CPU/CUDA RNG，并把该次保存的激活暂存
   CPU。普通批次继续 FP16，损失输入保持原来的 FP32，GradScaler 保持启用。

FP32 重算仍异常会明确报错。此修复没有定位首次产生非有限数值的具体模型算子，
也没有承诺整次训练不会再出现其它错误；重算次数和开销会写入日志。

训练条件保持为：GPU 0/1、GLOBAL batch 32（每卡 16）、120 轮、seed 0、EMA/AMP、
基础/验证输入 640、原多尺度和增强。AdamW 主 LR 0.0004、骨干 0.0002，warmup 1000，
flat 60，最后 8 轮无强增强，policy=[4,60,112]、MixUp=[4,60]。使用相同 9,449 张
训练图和 4,096 张验证图，六类 COCO bbox 评估，无坐标平移校准。
原始 b32 配置保留；新实验的数值策略明确记录为 float16、FP32 loss=True、FP32 retry=True。

## 更新及从 COCO 启动

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
CONFIG=configs/deim_dfine/deim_hgnetv2_s_sardet10pct_baseline_120e_b32_fp16_retry.yml
OUT=$(mktemp -d /root/zxy/experiments/deim_s_sardet10pct_baseline120e_b32_fp16retry_seed0_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_XXXXXX)
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

若服务器 GitHub HTTPS 仍超时，可从开发端为本次提交创建 Git bundle，通过 SSH 传输后
在服务器 `git fetch /tmp/本次.bundle main`、`git merge --ff-only FETCH_HEAD`。
这是 Git 历史的快进更新，保留工作区原有无关文件，不直接覆盖服务器源码。
子集已经生成，无需重新准备。

## 日志和指标

重连时激活 deim_zxy，并明确设置本次新的 OUT：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/本次新训练目录
export OUT
tail -f "$OUT/console.log"
```

另一个终端：

```bash
OUT=/root/zxy/experiments/本次新训练目录
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
watch -n 2 nvidia-smi
grep 'AMP_FP32_RETRY' "$OUT/console.log"
```

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/本次新训练目录
export OUT
python - <<'PY'
import json, os, pathlib
p=pathlib.Path(os.environ['OUT'])
rows=[json.loads(s) for s in (p/'log.txt').read_text().splitlines() if s.strip()]
for r in rows[-5:]:
    print('epoch',r['epoch'],'AP/AP50/AP75',
          [round(x*100,3) for x in r['test_coco_eval_bbox'][:3]],
          'fp32_retry_rate',r.get('train_fp32_retry_rate'))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

AP 在首轮评估后写入 log.txt，TensorBoard 事件在 summary/，最近一次 COCO 评估在
eval/latest.pth。检查点 best_stg1/best_stg2 要按实际评估 AP 比较。

## 后续中断时的恢复和检查点评估

本次执行的是重新初始化。以下仅供这次新训练今后中断时恢复；不要与正在运行的
任务同时执行。last.pth 在 epoch 112 前更新，不代表最终轮。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/本次新训练目录
LOG="$OUT/resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).console.log"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/last.pth" \
  --seed 0 --use-amp --output-dir "$OUT" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
tail -f "$LOG"
```

完整验证的 best_stg1/best_stg2 评估命令与
[120e/b32 交接文档](sardet_subset_baseline_120e_b32.md#检查点评估)相同：设置新 OUT，
使用其 resolved_config.yml 和 -r；每个候选写入新的评估目录。该命令评估官方
10,492 张验证图，训练过程继续使用固定 4,096 张筛选验证图。
