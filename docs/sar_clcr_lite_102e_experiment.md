# SAR-AIRcraft CLCR_LITE 单模块：102 轮训练

## 配置与协议

新增配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_clcr_lite_102e_b28_old_server.yml`。
直接继承旧服务器的冻结 102 轮 baseline，采用原版 CLCR-Lite：8 维输入、
Linear(8,16) → SiLU → Linear(16,1)，四层 decoder 与各类别共享 162 个新增参数。
输出层零初始化、beta=1，输入梯度贯通，从 epoch 0 启用。
实现说明见 `docs/sar_clcr_lite_experiment.md`。

| 项目 | 本次设置 |
| --- | --- |
| 模型与数据 | DEIM-D-FINE-M / HGNetv2-B2；SAR-AIRcraft 7 类，ID 0–6；train3047 / val442 / test879 |
| 初始化 | `/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth`，`train.py -t`，新建优化器 |
| 轮数与随机种子 | 102 轮，日志 epoch 0–101；seed 0 |
| GPU 与 batch | GPU 0、1；训练全局 batch 28，每卡 14；验证全局 batch 64 |
| 输入 | 基础与验证 640×640；训练多尺度 480–800；epoch 90 起固定 640 |
| 优化器 | AdamW，主 LR 3.5e-4；backbone 非 BN LR 3.5e-5；WD 1e-4，Norm/BN WD=0 |
| 学习率调度 | flatcosine；warmup 2000 steps；flat_epoch 49；no_aug_epoch 12；lr_gamma 0.5 |
| 增强 | policy [4,49,90]；Mosaic/MixUp 在 4≤epoch<49 使用；PhotometricDistort 在 4≤epoch<90 使用；水平翻转；关闭 RandomZoomOut/RandomIoUCrop |
| EMA 与精度 | EMA decay 0.9999、warmups 1000；AMP FP16、FP32 loss/retry；SyncBN、clip norm 0.1 |
| 评估 | 每轮使用 val.json、固定 640、EMA 权重、原 query×class 展平 top300 后处理；按最佳验证 AP 选择权重后评估 test.json |

训练与验证路径沿用 `/home/zxy/sar/datasets/SAR-AIRcraft-1.0` 及其 `COCO/` 标注，
无需重新准备数据。58 轮实验与 baseline 配置继续保留。
启动器归档完整解析配置、Git revision、工作树状态、seed、batch、输入、学习率、
增强、EMA/AMP、初始化路径与评估协议，供论文实验溯源。

## 服务器更新与启动

先查看服务器工作树；若存在改动或分叉，先保留并处理，再执行更新，不强制覆盖。

```bash
cd /home/zxy/sar/repos/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_clcr_lite_102e_b28_old_server.yml
```

启动器在 `EXPERIMENT_ROOT` 下创建新目录并打印 `STARTED OUT=... PID=... LOG=...`。
实际执行 `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py`，
通过 `nohup bash "$COMMAND" > "$LOG" 2>&1 < /dev/null &` 后台启动，
先保存 `$!` 到 `launcher.pid` 再 `disown`。控制台输出为 `console.log`。

## 日志、GPU、进程与指标

重连后重新激活环境，把 OUT 替换为启动器打印的实际目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际CLCR_LITE_102e运行目录
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
nvidia-smi -i 0,1
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

退出日志跟踪后，输出最佳验证与最近一轮的百分数指标：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(line) for line in (Path(sys.argv[1]) / 'log.txt').read_text().splitlines() if line.startswith('{')]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
for label, row in [('best_val', best), ('last_logged', rows[-1])]:
    metric = row['test_coco_eval_bbox']
    print(label, 'epoch=', row['epoch'],
          'AP/AP50/AP75/APs/APm/APl=', [round(v * 100, 4) for v in metric[:6]],
          'beta=', row.get('train_clcr_lite_beta'))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

逐轮 JSONL 为 `log.txt`，TensorBoard 为 `summary/`，COCO 评估数据为 `eval/latest.pth`。
`test_coco_eval_bbox` 的索引 0、1、2 分别为 AP@[.50:.95]、AP50、AP75，原值为 0–1。

## 中断后的恢复

原进程结束后，使用实际的原运行目录恢复：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原CLCR_LITE_102e运行目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
tail -f "$(ls -t "$OUT"/resume_*.console.log | head -n 1)"
```

恢复使用原 `resolved_config.yml` 与 `-r last.pth`，不传 `-t`；
控制台输出保存到新的 `resume_*.console.log`。
当前 solver 仅在 epoch<90 更新 `last.pth`，正常最后更新于 epoch 89。
进入最后 12 轮后发生中断时，此命令从该 checkpoint 重跑收尾，`last.pth` 不是最终轮权重。

## 选择最佳验证权重并评估测试集

训练结束后，按日志的最高验证 AP 选择对应阶段的权重；不根据阶段名称直接认定优劣。

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际CLCR_LITE_102e运行目录
CKPT=$(python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
import yaml

run = Path(sys.argv[1])
cfg = yaml.safe_load((run / 'resolved_config.yml').read_text())
rows = [json.loads(line) for line in (run / 'log.txt').read_text().splitlines() if line.startswith('{')]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
stop = cfg['train_dataloader']['collate_fn']['stop_epoch']
checkpoint = run / ('best_stg2.pth' if best['epoch'] >= stop else 'best_stg1.pth')
print(checkpoint)
print(f"val epoch={best['epoch']} AP={best['test_coco_eval_bbox'][0] * 100:.4f}", file=sys.stderr)
PY
)
EVAL=$(mktemp -d "$OUT/test_eval_XXXXXX")
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$CKPT" --test-only --seed 0 --use-amp \
  --output-dir "$EVAL" --summary-dir "$EVAL/summary" \
  -u val_dataloader.dataset.ann_file=/home/zxy/sar/datasets/SAR-AIRcraft-1.0/COCO/test.json \
  > "$EVAL/console.log" 2>&1 < /dev/null &
EVAL_PID=$!
printf '%s\n' "$EVAL_PID" > "$EVAL/launcher.pid"
disown "$EVAL_PID"
printf 'EVAL=%s PID=%s CHECKPOINT=%s\n' "$EVAL" "$EVAL_PID" "$CKPT"
tail -f "$EVAL/console.log"
```

独立测试指标见评估控制台，COCO 数据为评估目录的 `eval.pth`。

## 本地验证及范围

在 Conda `pytorch` 环境使用仓库 `load_config` 解析新配置，与冻结的 102 轮 baseline
逐字段比较：除实验目录、协议文档和四个 CLCR-Lite 显式设置外完全一致。
核对 102 轮、全局 batch 28、warmup 2000、flat 49、no_aug 12、policy [4,49,90]、
mixup [4,49]、stop 90、服务器数据路径和 EMA/AMP 设置。
本次仅增加配置与交接文档，未修改模型或训练器；本地未执行训练。
开发环境无法直连服务器，实际后台训练需在服务器执行上面的启动命令。
