# SAR-AIRcraft WFDR 单模块：102 轮训练

## 配置与协议

新增配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_wfdr_102e_b28_old_server.yml`。
直接继承旧服务器冻结的 102 轮 baseline，仅增加 W-FDR 训练损失。
W-FDR 的实现及定义见 `docs/sar_wfdr_experiment.md`。
沿用 58 轮单模块的 `use_wfdr=True`、`wfdr_weight=0.5` 和 `weight_dict.loss_wfdr=0.15`，
保留原 FGL 外部权重 0.15，合计为 `0.15 * (DFL + 0.5 * W1)`。
W1 使用原非均匀桶网格和 GT 双点插值，作用于最终层、aux 与 DN 的 FDR 分布，
原教师 DDF 蒸馏继续保留；不增加推理参数或计算。

| 项目 | 本次设置 |
| --- | --- |
| 模型与数据 | DEIM-D-FINE-M / HGNetv2-B2；7 类 ID 0–6；train3047 / val442 / test879 |
| COCO 初始化 | `/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth`，`train.py -t`，新建优化器 |
| 轮数与 seed | 102 轮，日志 epoch 0–101；seed 0 |
| GPU 与 batch | GPU 0、1；全局训练 batch 28，每卡 14；全局验证 batch 64 |
| 输入 | 基础及验证 640×640；训练多尺度 480–800；epoch 90 起固定 640 |
| 优化器 | AdamW，主 LR 3.5e-4、backbone 非 BN LR 3.5e-5；WD 1e-4，Norm/BN WD=0 |
| 调度 | flatcosine；warmup 2000 steps；flat_epoch 49；no_aug_epoch 12；lr_gamma 0.5 |
| 增强 | policy [4,49,90]；Mosaic/MixUp 在 4≤epoch<49 使用；PhotometricDistort 在 4≤epoch<90 使用；水平翻转；关闭 RandomZoomOut/RandomIoUCrop |
| EMA 与精度 | EMA decay 0.9999、warmups 1000；SyncBN、clip norm 0.1；AMP FP16、FP32 loss/retry |
| 评估 | 每轮 val.json、固定 640、EMA 权重、原 query×class 展平 top300 后处理；按最佳验证 AP 选择权重后评估 test.json |

数据沿用 `/home/zxy/sar/datasets/SAR-AIRcraft-1.0` 和其 `COCO/` 标注，无需重新准备。
58 轮实验和原 baseline 配置保留。启动器归档解析配置、Git revision、工作树状态、
seed、batch、输入、优化器、学习率、增强、EMA/AMP、初始化路径及评估协议。

## 服务器更新与双卡启动

若服务器工作树有改动或分叉，先检查并保留，再完成更新，不强制覆盖。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_wfdr_102e_b28_old_server.yml
```

启动器创建新目录并打印 `STARTED OUT=... PID=... LOG=...`。
实际使用 `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py`，
通过 `nohup bash "$COMMAND" > "$LOG" 2>&1 < /dev/null &` 启动后台训练，
保存 `$!` 到 `launcher.pid` 后 `disown`。首次日志为 `console.log`。
现有启动器要求 GPU 0、1 同时空闲；不空闲时不启动、不排队、不停止已有进程。

## 日志、GPU、进程、指标与 TensorBoard

重连后重新激活环境，显式设置 OUT 为启动器打印的实际 WFDR 目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际WFDR_102e运行目录
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
nvidia-smi -i 0,1
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C 退出日志跟踪后，查看最佳验证 AP 对应轮次、最近指标和 W-FDR 损失：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(line) for line in (Path(sys.argv[1]) / 'log.txt').read_text().splitlines() if line.startswith('{')]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
for label, row in [('best_val', best), ('last_logged', rows[-1])]:
    print(label, 'epoch=', row['epoch'], 'AP/AP50/AP75/APs/APm/APl=',
          [round(value * 100, 4) for value in row['test_coco_eval_bbox'][:6]])
    print({key: value for key, value in row.items()
           if key in ('train_loss_fgl', 'train_loss_ddf') or key.startswith('train_loss_wfdr')})
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

JSONL 指标在 `log.txt`，TensorBoard 事件在 `summary/`，训练 COCO 数据在 `eval/latest.pth`。
COCO bbox 索引 0、1、2 为 AP@[.50:.95]、AP50、AP75，原值 0–1，报告百分数乘 100。

## 中断恢复

原进程结束后，使用原目录及保存配置恢复：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原WFDR_102e运行目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
tail -f "$(ls -t "$OUT"/resume_*.console.log | head -n 1)"
```

恢复使用 `resolved_config.yml` 与 `-r last.pth`，不传 `-t`，日志为新的 `resume_*.console.log`。
当前 solver 只在 epoch<90 更新 `last.pth`，正常最后保存于 epoch 89；
收尾阶段中断后从该 checkpoint 重跑收尾，`last.pth` 不是最终轮权重。

## 按最佳验证 AP 选择权重并评估测试集

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际WFDR_102e运行目录
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
print(run / ('best_stg2.pth' if best['epoch'] >= stop else 'best_stg1.pth'))
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

依据实际验证 AP 选择 `best_stg1.pth` 或 `best_stg2.pth`，不凭阶段名称选择。
测试指标见评估控制台，COCO 数据保存为评估目录的 `eval.pth`。

## 本地验证与范围

Conda `pytorch` 环境下使用仓库 `load_config` 解析新配置，核对 W-FDR 开关与损失权重
与原 58 轮单模块一致；去除 W-FDR 设置、实验目录及协议文档字段后，
解析配置与冻结 102 轮 baseline 完全一致，包括学习率、增强、EMA/AMP、数据和评估设置。
本次仅新增配置和交接文档，未修改模型、损失实现或训练器，未执行本地或服务器训练。
开发环境无法直连服务器，实际训练需在服务器执行上述启动命令。
