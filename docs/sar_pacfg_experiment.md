# SAR-AIRcraft：DFGF 启发的 PACFG 适配实验

参考 [SPJFNet 的 DFGF 章节](https://arxiv.org/html/2508.04041)和本次提供的 PACFG 频域增强描述独立实现。
作者[官方仓库](https://github.com/bywlzts/SPJFNet)可检索，但本次没有取得可核对的 PACFG 改造源码。
因此这是检测任务的自有适配版，不宣称复现原 PACFG，也不包含贴文混入的 SCG-InfSA 图传播机制。

实现文件：`engine/deim/phase_aligned_frequency.py`。
输入 / 输出保持 `[B,C,H,W]`，接在 HybridEncoder 的 stride-8、256 通道投影特征上，FPN 融合之前。

## 本次实现选择

- 用 1×1 投影 / GroupNorm / SiLU 压到 64 通道，并进行一级 Haar 分解。
- LL 内部由深度卷积和点卷积生成同位置先验，不依赖亮图、其他传感器、Gamma 提亮或恢复监督。
- LL 和先验做 FP32 FFT，通过共轭乘积计算圆周相位一致性，避免在零幅值处直接求可微 angle。
- 点卷积门控接收两者的 log 幅值与相位余弦一致性；以幅值差校正增益，以相位正弦差引导旋转。
  增益范围为 exp(-0.5)..exp(0.5)，单次相位步长不超过 π/4；点卷积频域门控保持实信号的共轭对称关系。
- 原 LL、增强 LL 和三个高频幅值共同生成子带门控，高频系数乘以 0.5..1.5 的权重，保留系数符号。
- IDWT 重建后叠加轻量空间卷积分支，再投影回 256 通道；输出为 `X + alpha * delta`，alpha 初始为零。
- 整个适配器在 AMP 下使用 FP32，再还原输入 dtype，支持多尺度中的非二次幂尺寸与奇数边长。

这里的相位是中间空间特征的 Fourier 相位，不是 SAR 原始复数回波相位。
检测头、标签和原损失保持不变，不引入图像恢复或先验一致性辅助 loss。
日志增加 `train_pacfg_alpha`，运行元数据保存 PACFG 开关和瓶颈宽度。

配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pacfg_58e_b28_old_server.yml`。
只继承旧服务器的 58 轮 baseline；不叠加 REM、CrossGate-DBDM 或其他候选模块。
协议仍为 DEIM-D-FINE-M / HGNetv2-B2、七类 0–6、train3047 / val442 / test879、seed0、GPU0/1、
全局 batch28、基础 / 验证尺寸640、58轮、warmup432、flat29、no_aug8、lr_gamma0.5；
增强 policy [4,29,50]、mixup [4,29]、stop50；AdamW 主 LR3.5e-4、骨干非BN LR3.5e-5、WD1e-4。
原多尺度 / Mosaic / MixUp、EMA、AMP、FP32 loss / retry 和评估协议完整继承。
使用 COCO M 权重 `-t` 初始化；启动器归档解析配置、Git revision、seed、全部训练参数、权重及启动命令。
数据已经准备完成，无需重新处理。

## 服务器更新与双卡启动

Git 有未提交改动或分叉时先检查并保留现有文件，不要强制覆盖。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pacfg_58e_b28_old_server.yml
```

启动器新建实验目录，使用 `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2`，
以 `nohup ... > console.log 2>&1 < /dev/null &` 启动，先保存 `$!` 到 `launcher.pid`，再 `disown`。
本地无法直连服务器，没有远程训练或 AP 结果。

## 日志、指标、TensorBoard 和恢复

重连后先激活环境并重设 OUT：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际PACFG训练目录
tail -f "$OUT/console.log"
```

同一终端 Ctrl+C 退出日志跟踪后，分别运行：

```bash
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

指标为 `log.txt`，TensorBoard 为 `summary/`，训练 COCO 评估数据为 `eval/latest.pth`。
`test_coco_eval_bbox` 第0、1、2项是 val AP / AP50 / AP75，报告时乘100。

仅在训练进程已停止时恢复：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原PACFG训练目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复使用保存的解析配置和 `-r last.pth`，不同时使用 `-t`；跟踪打印的 `resume_*.console.log`。
当前 `last.pth` 正常最后更新于 epoch49，不是最终权重。

## 按 val AP 选权重，再做 test 评估

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原PACFG训练目录

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

权重按 val AP 选择，不按阶段文件名或 test AP 重选。独立评估数据为 `$EVAL/eval.pth`，
控制台输出 test AP / AP50 / AP75 / AR100。

## 本地定向验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_phase_aligned_frequency.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sar_module_integration.py -v
```

验证只针对已知信号的幅值 / 相位校正、零频谱梯度、奇数尺寸和 CPU autocast、实际 Encoder 路由与配置继承。
结果：PACFG 定向验证5项、现有模块回归8项全部通过；Python / Bash 及交接命令语法检查通过。
未加载真实图片或预训练权重，未启动完整训练；CUDA FP16 / 双卡和检测收益由服务器实验确认。
