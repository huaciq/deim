# SAR-AIRcraft：PGD-REM 单模块实验

本实现按用户指定的先验门控方向小波方案，把 LL 先验门控和 REM 方向卷积放进
同一次 Haar 分解/重建。它是新实验分支，已有 REM、PACFG、串联及分层配置保持原样。

## 确切计算路径

插入位置：HGNetv2-B2 的 stride-8 特征经过原 `input_proj` 投影为 256 通道之后、
HybridEncoder 的 FPN/PAN 融合之前。640 输入对应 `X=[B,256,80,80]`。

```text
X [B,256,H,W] -- Haar DWT（一次） --> LL, LH, HL, HH [B,256,ceil(H/2),ceil(W/2)]
  |                                  |
  |                                  +-- LL_base = Conv1x1_256->256(LL)
  |                                  |
  |                                  +-- U = Conv1x1_256->64(LL)
  |                                      P = DWConv5x5_64->64(U)
  |                                      gamma = 0.5 + sigmoid(Conv1x1_128->768([U, P-U]))
  |                                      分成 gamma_LH / gamma_HL / gamma_HH，各256通道
  |                                  |
  |                                  +-- F_LH = REM原LH链(LH) -- × gamma_LH --+
  |                                  +-- F_HL = REM原HL链(HL) -- × gamma_HL --+-- IDWT（一次） --> R
  |                                  +-- F_HH = REM原HH链(HH) -- × gamma_HH --+
  |                                      LL_base ---------------------------+
  +--------------------------------------------------------------------------> Y = X + alpha * R
```

- 先验旁路采用规格允许的 **5×5 深度卷积版本**；没有加入 FFT 或第二套小波。
- 仅旁路压缩到64通道；LL主路、三个方向分支、门控输出、逆变换及残差均保留256通道。
- `LL_base` 是普通1×1卷积，严格对应给定公式，没有额外残差系数。
- 高频链完整沿用现有 REM：LH 为 `1×1 -> 1×3 -> 3×1 -> 3×3`，
  HL 为 `1×1 -> 3×1 -> 1×3 -> 3×3`，HH 为 `1×1 -> 3×3`；
  每个卷积单元保留原来的 BN 和 SiLU。
- 先验深度卷积以5×5均值核初始化，使用复制填充，并参与训练。
  `U` 与 `P-U` 拼接提供压缩 LL 与平滑先验的显式比较。
  三组门控均为逐空间、逐通道的 `[0.5,1.5]` 调制值。
- 沿用带符号、逐通道的正交 Haar；奇数尺寸在右/下复制填充，IDWT后裁回原尺寸。
- 只有一个可学习标量 `alpha`，初始化为0，初始输出严格等于X。
  第一次反传可更新alpha，随后梯度进入方向链和先验门控；没有第二个零初始化残差头。
- 适配器在原网络初始化之后、隔离的随机数上下文内创建，保留同seed下原网络参数
  及后续decoder/数据随机数序列。原损失、FDR、LQE和后处理均保持原样。

256通道、64通道先验配置下，PGD-REM有2,940,545个可训练参数，
其中先验旁路117,120个；原REM为2,823,681个，总增量116,864个。

这里的5×5卷积是在LL上的局部结构先验，不是整幅图的全局频谱算子。
门控通过检测损失学习，没有GT轮廓掩码或预先规定的旁瓣标签。
“旁瓣抑制、缓解+2.75%框宽偏大”是本实验待检验的解释；AP75提升本身不能证明该因果关系。

## 文件与固定训练协议

- `engine/deim/radar_enhancement.py`：新增 `LowFrequencyStructureGate` 和
  `PriorGatedDirectionalREM`，复用已有REM方向链和单alpha。
- `engine/deim/hybrid_encoder.py`：新增 `use_pgd_rem=False`、
  `pgd_rem_prior_channels=64`，启用时处理stride-8投影特征。
- `engine/solver/det_engine.py`：将alpha写入训练指标 `train_pgd_rem_alpha`。
- `tools/run_sar_module_experiment.sh`：在运行元数据中记录PGD-REM开关及旁路宽度。
- `configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pgd_rem_58e_b28_old_server.yml`：
  仅覆盖新模块、输出目录、本文档；继承完整旧服务器58轮baseline。
- `tests/test_pgd_rem.py`：针对计算路径、梯度、接入位置和实验协议的功能验证。

协议：DEIM-D-FINE-M / HGNetv2-B2、7类且ID为0–6、train3047 / val442 / test879、
seed0、GPU0/1、全局batch28（每卡14）、58轮、基础及验证尺寸640、训练多尺度480–800。
warmup432、flat29、no_aug8、lr_gamma0.5；增强policy [4,29,50]、mixup [4,29]、stop50。
AdamW主LR3.5e-4、骨干非BN LR3.5e-5、WD1e-4；Mosaic/MixUp、EMA、AMP FP16、
FP32 loss/retry及原评估协议全部继承。验证后处理仍为query×class展平top300。

新实验从同一COCO M权重通过 `-t` 初始化，不继承REM或PACFG训练权重；
恢复原实验才使用 `-r`。原数据已准备好，无需改标注或重新转换数据。
启动器保存完整解析配置、源配置、Git revision、seed、优化器/增强/EMA/AMP配置、
初始化路径和执行命令。不要与旧102轮结果直接计算模块增益。

## 服务器更新与启动

先查看状态；若有未提交修改或分叉，保留并检查，不执行强制覆盖。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pgd_rem_58e_b28_old_server.yml
```

启动器创建新实验目录，使用 `CUDA_VISIBLE_DEVICES=0,1` 和
`torchrun --standalone --nproc_per_node=2 train.py`。实际启动为
`nohup bash "$COMMAND" > "$LOG" 2>&1 < /dev/null &`，随即保存 `$!` 至
`launcher.pid` 后再 `disown`。首次日志为 `console.log`；不会覆盖已有运行目录。

## 日志、指标与恢复

重连后显式激活环境，并将OUT替换为启动器输出的实际目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际PGD_REM运行目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C退出日志跟踪后，读取最佳验证指标与alpha：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(line) for line in (Path(sys.argv[1]) / 'log.txt').read_text().splitlines() if line.startswith('{')]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
for label, row in [('best_val', best), ('last_logged', rows[-1])]:
    print(label, 'epoch=', row['epoch'],
          'AP/AP50/AP75=', [round(v * 100, 4) for v in row['test_coco_eval_bbox'][:3]],
          'pgd_rem_alpha=', row.get('train_pgd_rem_alpha'))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

`log.txt`是JSONL epoch指标；`test_coco_eval_bbox`第0/1/2项分别为AP/AP50/AP75，
值在0–1，报告百分数乘100。TensorBoard事件在 `summary/`，COCO数据在 `eval/latest.pth`。

原进程停止后恢复同一实验：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原PGD_REM运行目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复加载归档的 `resolved_config.yml` 和 `last.pth`，使用 `-r`，不同时传 `-t`。
恢复日志为启动器打印的 `resume_*.console.log`。当前solver的 `last.pth` 仅在
无增强阶段之前更新，本协议正常最后更新于epoch49，不能当作最终epoch权重。

## 按验证AP选择权重并评估测试集

候选确定后按日志中实际最高验证AP选择阶段权重，不单凭阶段文件名选取：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原PGD_REM运行目录
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

独立测试结果在 `$EVAL/eval.pth`，指标见对应控制台日志。

## 本地验证与边界

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_pgd_rem.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_radar_enhancement.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_rem_pacfg_combination.py -v
```

验证覆盖256通道单次DWT/IDWT、方向链复用、三组门控的解析重建、先验差异响应、
奇偶尺寸、零alpha首次梯度、后续所有分支梯度、CPU/CPU BF16 autocast、
baseline初始化/权重接入、stride-8进入FPN，以及新配置只改变模块和归档路径。
结果：PGD-REM 6项、原REM 5项、REM/PACFG串联与分层回归5项，合计16项全部通过。
启动器、本文5段Bash命令及其中的Python片段语法检查通过，未执行服务器命令。
本地不含真实数据集、COCO权重或双4090训练；显存、耗时和检测收益需服务器实验确认。
