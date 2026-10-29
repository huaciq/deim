# SAR-AIRcraft CrossGate-DBDM：58 轮单模块实验

按本次提供的 CrossGate-DBDM 改造设计实现，不宣称复现 DBCFS-Net 作者代码。
模块在 HGNetv2-B2 的第二个 stage 替换 stride 4→8 下采样，保持原有通道数、步长和三个输出尺度。
配置只继承旧服务器 58 轮 baseline，不叠加 REM、DSG、SCS、SLQC 或 MPQ。

实现：`engine/backbone/crossgate_dbdm.py`。每个局部 2×2 块取四个相位，分别预测二维 offset；
原特征通道均值投影的 Sobel 边缘能量同时进入 offset 预测和融合门控。
offset 用 tanh 限制在输入特征 ±1 像素，采用双线性采样，四路拼接后以 1×1 卷积融合。
门控根据动态融合特征、原下采样特征和下采样边缘图生成通道权重。
输出为 `residual + res_scale * gate * (dynamic - residual)`；
offset 预测末层和 res_scale 初始为零，其余分支正常初始化。
奇数尺寸输出采用原 3×3/stride-2/pad-1 的向上取整规则。
边缘能量与采样坐标计算使用 FP32，融合卷积保留现有 AMP 行为。

原下采样的 `conv` / `bn` / `act` / `lab` 子模块和参数名保留，COCO 初始化继续加载原权重。
新模块参数仍归入原有 backbone 优化器分组。日志增加 `train_crossgate_dbdm_res_scale`，
运行元数据记录模块开关；不引入额外标签、loss 或检测头修改。

配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_crossgate_dbdm_58e_b28_old_server.yml`。
固定协议：七类 0–6，train 3047 / val 442 / test 879；seed 0，GPU 0/1，全局 batch 28；
基础 / 验证尺寸 640，58 轮，warmup 432，flat 29、no_aug 8、lr_gamma 0.5；
增强 policy [4,29,50]、mixup [4,29]、stop 50；AdamW 主 LR 3.5e-4、骨干非 BN LR 3.5e-5、WD 1e-4；
保留多尺度 / Mosaic / MixUp、EMA、AMP 和现有 FP32 loss / retry；使用 COCO M 权重 `-t` 初始化。
启动器归档解析配置、Git revision、seed、全部实验参数、初始化路径与实际启动命令。
数据已准备完成，无需重新解压或转换。

## 旧服务器更新与启动

Git 存在未提交改动或分叉时，先查看并保留现有文件，不要强制覆盖。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_crossgate_dbdm_58e_b28_old_server.yml
```

启动器创建新目录，使用 `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2`，
以 `nohup ... > console.log 2>&1 < /dev/null &` 后台启动，先保存 `$!` 到 `launcher.pid`，再 `disown`。
本地无法直连服务器，没有远程执行或完整训练结果。

## 日志、指标和恢复

重连后重新激活环境、设置真实 OUT：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际CrossGate训练目录
tail -f "$OUT/console.log"
```

在同一终端 Ctrl+C 退出日志跟踪后，分别运行：

```bash
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

训练 JSONL 指标在 `log.txt`，TensorBoard 在 `summary/`，COCO 评估数据在 `eval/latest.pth`。
`test_coco_eval_bbox` 第 0、1、2 项为 val AP / AP50 / AP75，报告时乘 100。

只在训练进程已停止时恢复：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原CrossGate训练目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复使用保存的 `resolved_config.yml` 和 `-r last.pth`，不同时提供 `-t`；
跟踪启动器打印的 `resume_*.console.log`。当前 `last.pth` 正常最后更新于 epoch 49，不是最终权重。

## 按 val AP 选模型后的 test 评估

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原CrossGate训练目录

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

权重根据 val AP 选择，不能仅按 `best_stg1` / `best_stg2` 文件名判断，也不按 test AP 重选。
test AP / AP50 / AP75 / AR100 在控制台输出；独立评估数据保存为 `$EVAL/eval.pth`。

## 本地定向验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_crossgate_dbdm.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sar_module_integration.py -v
```

验证仅针对采样坐标与像素位移、Sobel 边缘、梯度、CPU autocast、骨干接入、原权重键和配置继承。
结果：CrossGate-DBDM 定向验证 6 项、现有模块回归 8 项全部通过；Python / Bash 和交接命令语法检查通过。
定向验证同时确认新增融合 BN 仍使用原零权重衰减分组，offset 参数使用原 backbone 学习率。
未使用真实数据或预训练权重，未启动完整训练；CUDA FP16 与训练收益由服务器实验确认。
