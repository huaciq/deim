# SAR-AIRcraft REM：58 轮单模块实验

参考 [MARSS（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_MARSS_Radar_Semantic_Segmentation_via_Modular_Attention_and_State_Space_CVPR_2026_paper.html)
第 3.1 节 / Figure 3 实现四子带卷积分支。固定、逐通道的 Haar DWT 分解后：
LL 用 1×1；LH 用 1×1 → 1×3 → 3×1 → 3×3；HL 交换轴向顺序；HH 用 1×1 → 3×3。
各卷积采用项目的 BN / SiLU。为适配 DEIM 的同尺寸特征，本实现以 IDWT 和残差合成输出，
取代原文图示的 concat；残差 alpha 初始为 0，其余分支正常初始化。
这是按论文设计实现的检测适配版，没有复制未经核实的第三方代码。

模块：`engine/deim/radar_enhancement.py`；插入在 HybridEncoder 的 stride-8、256 通道投影特征上，
位于 FPN 融合之前；输出尺寸和原有三个检测尺度保持一致。
通过 `HybridEncoder.use_rem` 开启，默认关闭；日志增加 `train_rem_alpha`，运行元数据记录 REM 开关。

实验配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_58e_b28_old_server.yml`。
它仅在旧服务器 58 轮 baseline 上增加 REM、独立输出路径和本文归档路径。
本次只提供这一组实验，不新增消融或组合；已有数据无需重新处理。

固定协议：DEIM-D-FINE-M / HGNetv2-B2，七类 0–6，train 3047 / val 442 / test 879；
seed 0，GPU 0/1，全局 batch 28，基础 / 验证尺寸 640，58 轮，warmup 432，flat 29、no_aug 8，
增强 policy [4,29,50]、mixup [4,29]、stop 50；AdamW LR 3.5e-4、骨干 3.5e-5、WD 1e-4，
lr_gamma 0.5，保留原多尺度 / Mosaic / MixUp、EMA、AMP 及 FP32 loss / retry 设置。
初始化为 COCO M 权重 `-t`；每轮 val 评估，按 val AP 选择权重，再报告 test。
启动器保存解析配置、Git revision、seed、全部训练参数、权重路径和实际启动命令。

## 旧服务器更新与双卡启动

Git 存在未提交改动或分叉时先检查并保留现有文件，不要强制覆盖。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_58e_b28_old_server.yml
```

启动器创建新的实验目录，使用 `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2`，
以 `nohup ... > console.log 2>&1 < /dev/null &` 后台运行，先保存 `$!` 到 `launcher.pid`，再 `disown`。
本地没有直连服务器，交付不代表已经启动或完成训练。

## 重连后的日志、指标和恢复

先把 OUT 设置为启动器打印的实际目录：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际REM训练目录

tail -f "$OUT/console.log"
```

Ctrl+C 退出日志跟踪后，可分别运行：

```bash
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

指标在 `log.txt`，TensorBoard 在 `summary/`，训练 COCO 评估数据在 `eval/latest.pth`。
`test_coco_eval_bbox` 的第 0、1、2 项是 val AP / AP50 / AP75，报告时乘 100。

只在训练进程已经停止时恢复：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原REM训练目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复使用 `resolved_config.yml` 和 `-r last.pth`，不会同时使用 `-t`；跟踪启动器打印的
`resume_*.console.log`。当前 `last.pth` 正常最后更新于 epoch 49，不是最终权重。

## 训练完成后的 test 评估

以下先按训练日志中的 val AP 选对应阶段最佳文件，再用双卡评估 test；不按阶段文件名或 test AP 选模型。

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原REM训练目录

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

控制台输出 test AP / AP50 / AP75 / AR100，本次独立评估数据保存为 `$EVAL/eval.pth`。

## 本地定向验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_radar_enhancement.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sar_module_integration.py -v
```

验证仅针对固定 Haar 系数与重建、奇数尺寸、梯度、初始残差、CPU autocast、Encoder 实际路由和配置继承。
结果：REM 定向验证 5 项、现有模块接入回归 8 项全部通过；修改的 Python、Bash 和交接命令语法检查通过。
没有使用真实数据或预训练权重，没有启动完整训练；CUDA FP16 和实验收益由服务器实际训练确认。
