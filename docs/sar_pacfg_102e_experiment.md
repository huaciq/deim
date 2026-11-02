# SAR-AIRcraft PACFG 单模块：新、旧服务器 102 轮训练

## 服务器选择与配置

本次用户明确选择新服务器。此后默认旧服务器，只有用户主动说明新服务器时才切换；
本次新服务器实验不改变该默认规则。两套配置的模型、损失及训练协议一致，仅服务器路径不同。

| 项目 | 旧服务器（后续默认） | 新服务器（本次使用） |
| --- | --- | --- |
| 配置文件后缀 | `pacfg_102e_b28_old_server.yml` | `pacfg_102e_b28_new_server.yml` |
| 仓库 | `/home/zxy/sar/repos/deim` | `/root/zxy/deim` |
| 数据 | `/home/zxy/sar/datasets/SAR-AIRcraft-1.0` | `/root/zxy/datasets/SAR-AIRcraft-1.0` |
| 实验根目录 | `/home/zxy/sar/experiments` | `/root/zxy/experiments` |
| 初始化权重 | `/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth` | `/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth` |

完整配置路径：

- `configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pacfg_102e_b28_old_server.yml`。
- `configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pacfg_102e_b28_new_server.yml`。

新服务器配置继承冻结的 102 轮 baseline；旧服务器配置继承新服务器 PACFG 配置，
仅覆盖数据与输出路径，确保两套方法参数同步。PACFG 沿用现有 58 轮单模块实现：
`use_pacfg=True`、`pacfg_bottleneck=64`、`pacfg_feat_index=0`，
接在 stride-8、256 通道投影后、FPN/PAN 前，残差 alpha 零初始化。
实现说明见 `docs/sar_pacfg_experiment.md`，训练日志记录 `train_pacfg_alpha`。

## 固定训练协议

| 项目 | 两套配置共用设置 |
| --- | --- |
| 模型与数据 | DEIM-D-FINE-M / HGNetv2-B2，4 层 decoder；7 类 ID 0–6；train3047 / val442 / test879 |
| 初始化 | 同一 COCO M 90e 权重，通过 `train.py -t` 初始化，新建优化器 |
| 轮数与 seed | 102 轮，日志 epoch 0–101；seed 0 |
| GPU 与 batch | GPU 0、1；全局训练 batch 28，每卡 14；全局验证 batch 64 |
| 输入 | 基础与验证 640×640；训练多尺度 480–800；epoch 90 起固定 640 |
| 优化器 | AdamW，主 LR 3.5e-4、backbone 非 BN LR 3.5e-5；WD 1e-4，Norm/BN WD=0 |
| 调度 | flatcosine；warmup 2000 steps；flat_epoch 49；no_aug_epoch 12；lr_gamma 0.5 |
| 增强 | policy [4,49,90]；Mosaic/MixUp 在 4≤epoch<49 使用；PhotometricDistort 在 4≤epoch<90 使用；水平翻转；关闭 RandomZoomOut/RandomIoUCrop |
| EMA 与精度 | EMA decay 0.9999、warmups 1000；SyncBN、clip norm 0.1；AMP FP16、FP32 loss/retry；PACFG 内部沿用 FP32 |
| 评估 | 每轮 val.json、固定 640、EMA 权重、原 query×class 展平 top300 后处理；按最佳验证 AP 选权重后评估 test.json |

沿用各服务器现有图像及 `COCO/` 标注，无需重新准备数据。原 58 轮实验和 baseline 保留。
启动器归档完整解析配置、Git revision、工作树状态、seed、batch、输入、学习率、增强、
EMA/AMP、初始化路径及评估协议。两服务器分别记录各自实际 revision 和运行目录。

## 本次：新服务器更新与启动

先查看工作树；有本地改动或分叉时先检查并保留，不强制覆盖。

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/root/zxy/experiments
export INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pacfg_102e_b28_new_server.yml
```

## 旧服务器更新与启动（后续默认）

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pacfg_102e_b28_old_server.yml
```

两套命令均创建带时间戳和随机后缀的新目录，使用
`CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py`。
启动器以 `nohup bash "$COMMAND" > "$LOG" 2>&1 < /dev/null &` 后台运行，
先保存 `$!` 到 `launcher.pid` 再 `disown`，打印 `STARTED OUT=... PID=... LOG=...`。
首次日志为 `console.log`。现有启动器要求两卡同时空闲，不空闲时不启动、不排队。

## 重连后的环境与目录

新服务器本次实验，重连后执行并替换 OUT：

```bash
cd /root/zxy/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export DATA_ROOT=/root/zxy/datasets/SAR-AIRcraft-1.0
export OUT=/root/zxy/experiments/替换为实际PACFG_102e运行目录
```

旧服务器实验，重连后执行并替换 OUT：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export DATA_ROOT=/home/zxy/sar/datasets/SAR-AIRcraft-1.0
export OUT=/home/zxy/sar/experiments/替换为实际PACFG_102e运行目录
```

## 日志、GPU、进程、指标与 TensorBoard

先执行对应服务器的重连环境命令，然后运行以下共用命令：

```bash
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
nvidia-smi -i 0,1
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C 退出日志跟踪后，查看最佳验证 AP 对应轮次、最近指标和 PACFG 残差系数：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(line) for line in (Path(sys.argv[1]) / 'log.txt').read_text().splitlines() if line.startswith('{')]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
for label, row in [('best_val', best), ('last_logged', rows[-1])]:
    print(label, 'epoch=', row['epoch'], 'AP/AP50/AP75/APs/APm/APl=',
          [round(value * 100, 4) for value in row['test_coco_eval_bbox'][:6]],
          'pacfg_alpha=', row.get('train_pacfg_alpha'))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

JSONL 为 `log.txt`，TensorBoard 为 `summary/`，训练 COCO 数据为 `eval/latest.pth`。
COCO bbox 索引 0、1、2 为 AP@[.50:.95]、AP50、AP75，原值 0–1，百分数报告乘 100。

## 中断恢复

原进程结束后，先执行对应服务器的重连环境命令，再恢复原目录：

```bash
bash tools/run_sar_module_experiment.sh --resume "$OUT"
tail -f "$(ls -t "$OUT"/resume_*.console.log | head -n 1)"
```

恢复使用原 `resolved_config.yml` 与 `-r last.pth`，不传 `-t`；日志为新的 `resume_*.console.log`。
当前 solver 只在 epoch<90 更新 `last.pth`，正常最后保存于 epoch 89；
收尾阶段中断后从该 checkpoint 重跑收尾，`last.pth` 不是最终轮权重。

## 按最佳验证 AP 选权重并评估测试集

训练结束后，先执行对应服务器的重连环境命令，再执行：

```bash
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
  -u "val_dataloader.dataset.ann_file=$DATA_ROOT/COCO/test.json" \
  > "$EVAL/console.log" 2>&1 < /dev/null &
EVAL_PID=$!
printf '%s\n' "$EVAL_PID" > "$EVAL/launcher.pid"
disown "$EVAL_PID"
printf 'EVAL=%s PID=%s CHECKPOINT=%s\n' "$EVAL" "$EVAL_PID" "$CKPT"
tail -f "$EVAL/console.log"
```

按实际验证 AP 选择 `best_stg1.pth` / `best_stg2.pth`，不凭阶段名称或 test AP 选择。
测试指标见评估控制台，COCO 数据为评估目录下的 `eval.pth`。

## 本地验证与范围

在 Conda `pytorch` 环境使用仓库 `load_config` 解析两套配置，分别与对应服务器的冻结
102 轮 baseline 比较：除 PACFG 设置、实验目录及协议文档字段外完全一致。
两套配置去除服务器数据和输出路径后完全相同；PACFG 设置与原 58 轮单模块一致，
其中显式固定原默认 `pacfg_feat_index=0`。核对各服务器数据路径和全部训练调度继承。
本次仅新增配置、交接文档及服务器默认规则，未修改模型或训练器，未执行训练。
开发环境无法直连服务器，实际后台训练需在对应服务器执行上述启动命令。
