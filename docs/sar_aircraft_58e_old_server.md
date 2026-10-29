# SAR-AIRcraft 58 轮：旧服务器双卡交接

沿用 [58 轮固定协议](sar_aircraft_deim_m_58e_protocol.md)。新配置
`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_58e_b28_old_server.yml`
只覆盖旧服务器的图片、标注和默认输出路径，完整继承原 58 轮训练设置。
原 102 轮和 58 轮配置不修改；数据已完成转换，不需要重新解压或准备。

## 固定设置与路径

- 仓库：`/home/zxy/sar/repos/deim`；Conda：`deim_zxy`。
- 数据：`/home/zxy/sar/datasets/SAR-AIRcraft-1.0`；图片路径指向数据集根目录，
  标注为 `COCO/train.json`、`COCO/val.json`，最终测试使用 `COCO/test.json`。
- 原划分：train 3047 / val 442 / test 879；七类，ID 0–6。
- DEIM-D-FINE-M / HGNetv2-B2；初始化权重：
  `/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth`，新训练使用 `-t`。
- GPU 0、1；全局 batch 28（每卡 14）；seed 0；58 轮；基础与验证输入 640。
- AdamW：主 LR 3.5e-4、骨干非 BN LR 3.5e-5、WD 1e-4；warmup 432 步，
  flat_epoch 29、no_aug_epoch 8、lr_gamma 0.5。
- 保留 Mosaic / MixUp、多尺度与光度增强；关闭 ZoomOut / IoUCrop；
  policy [4,29,50]、mixup [4,29]、stop_epoch 50；最后 8 轮固定 640。
- 保留 EMA、AMP FP16、FP32 loss 与现有 FP32 重算；每轮 val 评估，按 val AP 选权重后报告 test。

启动器自动从自身位置定位仓库，并通过 `conda info --base` 定位 Conda。
`EXPERIMENT_ROOT`、`INITIAL_WEIGHTS` 可覆盖新训练的输出根目录和初始化权重；
未设置时保留原新服务器默认值。数据统计从配置中训练标注所在目录的 `stats.json` 读取。
每次新训练创建新目录，保存解析配置、协议、Git revision、工作区状态、完整实验参数和启动命令。

## Git 更新与双卡启动

在旧服务器执行。若工作区存在改动或分支分叉，先查看具体差异并保留原文件；不要 reset 或强制切换。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_58e_b28_old_server.yml
```

启动器打印 `STARTED OUT=... PID=... LOG=...`；将真实 OUT 保存供后续命令使用。
它沿用双卡空闲检查，然后运行 `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2`，
以 `nohup ... > console.log 2>&1 < /dev/null &` 启动，先保存 `$!` 到 `launcher.pid` 再 `disown`。
此交接没有远程执行，也没有训练完成或 AP 结果。

## 重连、日志、进程与指标

重连后先重新设置实际 OUT 并激活环境：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为启动器打印的实际目录

tail -f "$OUT/console.log"
```

在同一终端按 Ctrl+C 退出日志跟踪后，查看进程、GPU 和最近指标：

```bash
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
nvidia-smi -i 0,1
tail -n 3 "$OUT/log.txt"

python - "$OUT/log.txt" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.startswith('{')]
for row in rows[-3:]:
    values = row['test_coco_eval_bbox']
    print('val epoch', row['epoch'], dict(zip(['AP', 'AP50', 'AP75', 'AR100'],
          [round(values[i] * 100, 2) for i in [0, 1, 2, 8]])))
PY

tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

训练 JSONL 指标为 `log.txt`，TensorBoard 为 `summary/`，训练中的 COCO 评估数据为
`eval/latest.pth`。指标索引 0、1、2 对应 AP@[.50:.95]、AP50、AP75，原始值乘 100 后报告。

## 恢复已停止的训练

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原训练目录

bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复使用保存的 `resolved_config.yml` 和 `-r "$OUT/last.pth"`，不同时提供 `-t`。
日志为启动器打印的 `resume_*.console.log`，可执行 `tail -f` 跟踪该准确路径。
当前 solver 仅在 epoch < 50 时更新 `last.pth`，正常最后一次为 epoch 49（从 0 开始）；
它不是最终权重。若在最后 8 轮中断，从该文件恢复会重跑收尾阶段。

## 按 val 选权重，再做 test 评估

训练完成后，先从每轮 val AP 选出对应的阶段最佳文件。
`best_stg2.pth` 仅在第二阶段保存过最佳结果时存在，不能仅凭阶段名称选择它。

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原训练目录

CKPT=$(python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

import yaml

run = Path(sys.argv[1])
cfg = yaml.safe_load((run / 'resolved_config.yml').read_text())
rows = [json.loads(line) for line in (run / 'log.txt').read_text().splitlines() if line.startswith('{')]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
stop = cfg['train_dataloader']['collate_fn']['stop_epoch']
stage = 'best_stg2' if best['epoch'] >= stop else 'best_stg1'
print(run / f'{stage}.pth')
print(f"Selected by val: epoch={best['epoch']} AP={best['test_coco_eval_bbox'][0] * 100:.4f}", file=sys.stderr)
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

本次独立评估数据为 `$EVAL/eval.pth`；控制台输出包含 test AP / AP50 / AP75 / AR100。
不按 test AP 重新挑选训练权重。研究协议的其余评估说明沿用原 58 轮文档。

## 本地验证与交付范围

本地只做 58 轮调度和配置继承的 CPU 定向回归，以及 Bash / 内嵌 Python 语法检查。
旧服务器子配置的解析结果与原 58 轮配置逐项一致，仅图片、标注和默认输出路径不同。
结果：CPU 定向回归 **5 项通过**；启动器与本文共 **6 段 Bash**、**5 段内嵌 Python** 通过语法检查。
定向回归可在 Windows PowerShell 复现：

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sar_58e_protocol.py -v
```

未加载真实数据或模型权重，未执行服务器训练；本地检查不能代替实际训练结果。
