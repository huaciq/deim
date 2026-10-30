# SAR-AIRcraft SPI-Module 实现与实验交接

## 实现与原 SPI 名称区分

新增 `engine/deim/scattering_physics.py::SPIModule`，由
`HybridEncoder.use_spi_module: True` 启用。仓库原有 `use_spi` / `ScatteringPatternIsolation`
保留其原含义；本方法使用独立的 `spi_modules`，不复用旧分解器、先验模板或辅助损失。

在 P3/P4/P5 的 input_proj 后、encoder transformer 前，各放一个独立的256通道模块，
输出尺寸及通道数不变。三个模块没有共享参数或BN统计。标准配置无P2注入；若启用已有P2/S2路径，
重构作用于该路径得到的投影特征，再进入原 transformer/FPN/PAN。
所有新权重在原 encoder 完成初始化后、隔离随机状态创建，保留共同权重及后续 decoder/数据随机序列。

严格采用给出的 PyTorch 代码结构：

1. `eroded=-max_pool2d(-X,3,1,1)`，`opened=max_pool2d(eroded,3,1,1)`，
   `peak=relu(X-opened)`。这是逐通道 min/max 池化实现，没有添加额外窗口尺度或软近似。
2. `mu=avg_pool2d(abs(X),3,1,1)`，`mean_square=avg_pool2d(X**2,3,1,1)`，
   `CoV=sqrt(relu(mean_square-mu**2)+1e-5)/(mu+1e-5)`。
   AvgPool沿用原示例默认的零填充及 count_include_pad=True，边界统计包含填充值。
3. peak、CoV分别经过 `GroupNorm(32,256)`，拼接后通过
   `Conv1x1(512→256,bias=False) → BatchNorm2d → Sigmoid` 得到空间/通道门控。
4. `salient=X*mask`，经两个并行 `DWConv1x5` 和 `DWConv5x1` 相加后，
   继续经过 `BatchNorm2d → SiLU(inplace=True) → Conv1x1(256→256,bias=False)`。
   两个条带卷积的 dilation=1，与提供的代码一致，不额外指定空洞率。
5. `Y=X+alpha*salient+beta*bridge`，alpha/beta均为 `(1,256,1,1)` 的零初始化可学习参数。

新增结构仅有启用开关；encoder路径固定k=3、eps=1e-5、GN组数32及条带尺寸5，
不添加损失权重、调度、手工注入比例或通道瓶颈。
每尺度201,728个参数，三个尺度共605,184个可训练参数；推理也执行这三个模块。
实际吞吐及显存开销需在服务器测量。

为避免AMP下X平方的FP16溢出，形态学、CoV及GroupNorm在FP32中计算。
门控和桥接CNN沿用训练autocast，残差计算返回输入dtype和尺寸。
没有clamp特征幅值、detach先验、NaN替换或数据相关的提前返回。
第一步alpha/beta即可获得梯度，更新后门控、归一化和桥接分支获得有效梯度。
恒等指初始前向；alpha/beta从第一步优化就开始改变，不表示整个第0轮始终恒等。
新增BN会按原规则更新自己的运行统计，不改变原BN的训练规则。

统计量来自学习特征，不直接等同于原始SAR强度/复数回波的散射分解。
固定3×3窗口在三个步长上覆盖的范围不同，不据此保证对应特定物理部件。
散射点连通、噪声抑制和检测增益是待验证机制。

## 文件、配置与固定协议

- `engine/deim/scattering_physics.py`：SPIModule完整结构、FP32先验及两路通道残差。
- `engine/deim/hybrid_encoder.py`、`engine/deim/__init__.py`：三尺度接入及导出类。
- `engine/solver/det_engine.py`：逐尺度记录alpha/beta的平均绝对值。
- `tools/run_sar_module_experiment.sh`：归档开关、通道/窗口/条带/初始化及插入位置。
- `tests/test_spi_module.py`：统计解析值、恒等/学习、路由、训练、恢复、EMA及部署验证。
- 单方法配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_spi_module_58e_b28_old_server.yml`。
- 方法一+二：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_spi_module_wfdr_58e_b28_old_server.yml`。

同一58轮协议：DEIM-D-FINE-M / HGNetv2-B2、7类ID0–6、train3047 / val442 / test879，
seed0、GPU0/1、全局batch28（每卡14）、基础/验证640、训练多尺度480–800；
warmup432、flat29、no_aug8、lr_gamma0.5；增强policy [4,29,50]、mixup [4,29]、stop50。
AdamW主LR3.5e-4、骨干非BN LR3.5e-5、WD1e-4，Mosaic/MixUp、EMA、AMP FP16、
FP32 loss/retry及query×class展平top300后处理继承。
组合配置额外继承W-FDR的 `0.15*(DFL+0.5*W1)`，原DDF继续保留。
两种实验均从同一COCO M权重 `-t` 开始。现有SAR-AIRcraft目录和0–6标注不变，无需重新准备数据。
启动器归档配置、Git revision、seed、batch、输入、优化器、增强、EMA/AMP、初始化及评估协议。

## 服务器更新及启动

若服务器工作树有改动或分叉，先检查并保留改动，不强制覆盖。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_spi_module_58e_b28_old_server.yml
```

启动器创建新目录，用 `CUDA_VISIBLE_DEVICES=0,1`、
`torchrun --standalone --nproc_per_node=2 train.py`；
执行 `nohup bash "$COMMAND" > "$LOG" 2>&1 < /dev/null &` 后，保存 `$!` 至launcher.pid再disown。
首次日志为console.log。前一实验结束后，组合实验另起新目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_spi_module_wfdr_58e_b28_old_server.yml
```

## 日志、指标、TensorBoard及恢复

重连后显式设置OUT为启动器打印的目录并激活环境：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际SPI-Module运行目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C退出日志跟踪后，读取最佳验证AP、通道残差强度和组合损失：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(line) for line in (Path(sys.argv[1]) / 'log.txt').read_text().splitlines() if line.startswith('{')]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
for label, row in [('best_val', best), ('last_logged', rows[-1])]:
    losses = {key:value for key,value in row.items() if key.startswith('train_spi_module_') or key.startswith('train_loss_wfdr')}
    print(label, 'epoch=', row['epoch'],
          'AP/AP50/AP75=', [round(value * 100, 4) for value in row['test_coco_eval_bbox'][:3]],
          'losses=', losses)
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

JSONL指标在log.txt；COCO bbox第0/1/2项是AP/AP50/AP75，百分数报告乘100。
TensorBoard在summary/，训练COCO数据在eval/latest.pth。
原进程停止后恢复同一SPI-Module实验：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原SPI-Module运行目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复用resolved_config.yml和 `-r last.pth`，不同时传 `-t`；日志为打印的resume_*.console.log。
本协议last.pth正常最后更新于epoch49，不是最后一轮权重。

## 按验证AP选择权重并评估测试集

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原SPI-Module运行目录
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

按实际验证AP选择best_stg1.pth/best_stg2.pth，独立评估数据在 `$EVAL/eval.pth`，指标见控制台。

## 本地功能验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_spi_module.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_wfdr.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sdcp.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_clcr_lite.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_clcr.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_query_prototypes.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_pgd_rem.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sar_58e_protocol.py -v
```

验证范围：孤立峰/连续亮区、CoV及填充边界、完整两路残差、正交桥接、FP32/CPU BF16 autocast、
FP16输入平方溢出的避免、零初始化恒等及两步梯度、三个尺度在transformer前的路由、
原权重/RNG/旧SPI兼容、真实检测损失与Lite/W-FDR组合、参数加载、EMA和部署转换。
训练日志会记录 train_spi_module_alpha_p3_mean_abs / train_spi_module_beta_p3_mean_abs，
P4/P5同理。它们是通道平均绝对值，不是单个共享标量。
结果：SPI-Module 12项、W-FDR 12项、SDCP 14项、Lite 11项、原CLCR 10项、
原decoder/MPQ回归7项、PGD-REM 6项、58轮协议5项，合计77项全部通过。
启动器、本文6段Bash命令及嵌入Python片段语法检查通过，没有执行服务器命令。
本地验证不包含真实数据训练或CUDA双卡吞吐/显存测试，不据此宣称检测增益。
