# SAR-AIRcraft：REM + PACFG 双模块消融

本次只增加组合配置，补齐 baseline / REM / PACFG / REM+PACFG 的四组消融。
沿用已训练的两个适配器，不改其内部结构；组合从同一 COCO M 权重重新初始化，
不接着 REM 或 PACFG 的训练权重继续训练。比较使用同一验证集和同一最佳 val AP 选择规则。

## 结构与研究假设

路径为 `HGNetv2 -> stride-8 投影 -> REM -> PACFG -> FPN/PAN -> D-FINE decoder`；
stride-16 / stride-32 分支沿用原路径。两者各自保留零初始化残差尺度，
PACFG 瓶颈为64。现有 HybridEncoder 已支持该顺序，无需修改模型代码。

- REM 对低频与三个方向高频分支分别做卷积，再进行同尺寸重建，侧重局部方向结构。
- PACFG 用内部先验校正 LL 的 Fourier 幅值和相位，再调制高频子带，侧重全局结构与跨频协调。
- 二者都包含小波处理，收益可能重叠或相互干扰；组合是否优于单模块由本次实验判断。

实现来源和适配差异见 [REM 文档](sar_rem_experiment.md)和 [PACFG 文档](sar_pacfg_experiment.md)。
原始研究为 [MARSS](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_MARSS_Radar_Semantic_Segmentation_via_Modular_Attention_and_State_Space_CVPR_2026_paper.html)
和 [SPJFNet / DFGF](https://arxiv.org/abs/2508.04041)。当前代码是检测适配实现；
PACFG 的 Fourier 相位指中间特征的空间结构，不是原始 SAR 复数回波相位。

论文主线目前是待验证的“局部方向结构提取 -> 全局频谱协调 -> 目标级判别”。
前两步对应现有模块；第三步只作为后续方向，尚未定义或实现新的模块。
先评估组合与最佳单模块的差异，再决定是否加入针对相似机型判别的任务模块。
用户提供的单模块验证结果中 AP_S 均低于 baseline，不能据此宣称全面改善小目标检测。
最终候选需补重复种子、逐类/尺度结果及计算开销，测试集不用于筛选组合或插入顺序。
论文需明确借鉴来源及实际改动，模块数量本身不能证明方法贡献。

## 固定协议

配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_pacfg_58e_b28_old_server.yml`。
继承旧服务器58轮 baseline：DEIM-D-FINE-M / HGNetv2-B2、七类0–6、
train3047 / val442 / test879、seed0、GPU0/1、全局batch28（每卡14）、
基础/验证尺寸640、训练多尺度480–800、58轮、warmup432、flat29、no_aug8、lr_gamma0.5；
增强policy [4,29,50]、mixup [4,29]、stop50；AdamW 主LR3.5e-4、骨干非BN LR3.5e-5、
WD1e-4。Mosaic / MixUp、EMA、AMP FP16、FP32 loss/retry、原损失和评估协议完整继承。
其他实验模块保持关闭。数据已经准备完成，无需再解压或转换。

启动器归档解析配置、Git revision、seed、全部训练参数、初始化权重、模块开关与启动命令。
日志同时记录 `train_rem_alpha` 和 `train_pacfg_alpha`。

## 服务器更新与训练

若工作区有改动或分叉，先检查并保留现有文件，不强制覆盖。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_pacfg_58e_b28_old_server.yml
```

启动器新建实验目录，以 `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2`
双卡运行，使用 `nohup ... > console.log 2>&1 < /dev/null &`，先保存 `$!` 到 `launcher.pid`，
再 `disown`。用 `-t` 加载 COCO M 权重。当前开发环境无法直连旧服务器，本地交付不代表已启动训练。

## 日志、指标和恢复

重连后激活环境并将 OUT 设置为启动器打印的实际目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际REM_PACFG训练目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C 退出日志跟踪后，可以启动 TensorBoard：

```bash
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

`log.txt` 中 `test_coco_eval_bbox` 是验证集指标：第0/1/2项为 AP/AP50/AP75，
第3/4/5项为 AP_S/AP_M/AP_L；乘100为百分数。训练评估数据在 `eval/latest.pth`。
只在训练进程已停止时恢复，跟踪启动器打印的 `resume_*.console.log`：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原REM_PACFG训练目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复使用保存的配置和 `-r last.pth`，不同时使用 `-t`。
当前 `last.pth` 正常最后更新于 epoch49，不是最终权重。

## 验证集选择权重与测试集评估

先用同一 val AP 规则比较四组；方法选择完成后再做测试集评估。
下面按日志中的最佳 val AP 选择对应阶段权重，不按阶段文件名假定其优劣：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原REM_PACFG训练目录

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

独立评估数据在 `$EVAL/eval.pth`，指标在评估控制台日志中。

## 本地定向验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_rem_pacfg_combination.py -v
```

验证四组初始化与协议一致、REM输出直接进入PACFG、两模块在CPU/CPU autocast下的梯度、
三个输出尺度和双残差尺度指标。结果：3项定向验证全部通过，Python及5段Bash交接命令的语法检查通过。
没有在本地启动真实训练；CUDA FP16、双卡显存、训练耗时与AP收益需服务器确认。
