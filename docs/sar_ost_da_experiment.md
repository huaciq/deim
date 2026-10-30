# SAR-AIRcraft OST-DA / TACA 实现与实验交接

## 结构与接口

按附件代码实现 `OrthogonalTopologyDeformAttn`，位于
`engine/deim/orthogonal_topology_attention.py`，通过
`DFINETransformer.use_ost_da: True` 替换各 TransformerDecoderLayer 的 cross_attn。
原 MSDeformableAttention 保留为默认实现，自注意力、原 gateway、FFN、FDR、LQE及后处理沿用原路径。

固定实验结构为 D=256、H=8、L=3、每层K=4：前4头自由二维偏移，后4头为拓扑头。
按附件可执行代码，每个Query的每个拓扑头独立预测一个theta，而不是四头共享一个角度：
`theta=pi*sigmoid(angle_proj(query))`，输出4个角。
`u_body=(cos(theta),sin(theta))`、`u_wing=(-sin(theta),cos(theta))`。
拓扑头原偏移层的两个输出解释为r及扰动参数，`delta=.25*tanh(raw_delta)`；
每层前2点为 `r*u_body+delta*u_wing`，后2点为 `r*u_wing+delta*u_body`。
不加入r截断、额外旋转标注、角度损失、对称性损失或新的残差缩放参数。

采样位置严格使用 `ref_xy + offset/K*ref_wh*.5`；不夹紧到框内或图像内。
原 Decoder 的 `[B,Q,1,4]` 共享参考框会广播到3个尺度，也支持附件的 `[B,Q,L,4]`。
角度方向在乘框宽高之前的局部偏移坐标中正交；长宽缩放后不保证在图像像素坐标中仍成90度。
扰动的数学开区间在浮点饱和时可达到±0.25，与tanh实现一致。

每头注意力权重在全部L*K=12个点上softmax。聚合结果按自由头→拓扑头拼接为256维，
经 `Linear(256,256) → LayerNorm(256) → Sigmoid` 得到256维门控，
`output=output_proj(aggregated*gate)`。保留附件的value_proj、output_proj及其bias。
门控是可学习通道交互，不等同于硬编码的双轴散射对称性检测。

当前D-FINE将value预先拆成各尺度的 `[B,H,D/H,H_l*W_l]`；模块会重组为 `[B,S,D]` 后再做
一次value_proj，不丢失level/head/channel顺序。也直接支持附件的扁平value输入。
复用原生 `deformable_attention_core_func`，使用bilinear、zeros padding、align_corners=False。
不使用discrete采样；若同时配置 `cross_attn_method: discrete` 会明确报错。
投影层沿用AMP；三角函数、偏移组装、softmax、网格与采样核在FP32计算，输出返回Query dtype。

初始化按附件：offset权重为0、bias为标准极坐标网格；attention权重/bias为0；
angle权重/bias为0，对应初始theta=pi/2。其余新增线性层保持PyTorch默认初始化，LayerNorm采用默认值。
替换在原decoder层完成初始化后、隔离随机状态进行，保留共同层权重及后续随机序列。
新增投影和门控会主动改变初始输出，不是恒等适配器。
三路Query条件投影的权重初始为零，因此首步对独立Query输入的直接梯度为零；
投影参数本身首步即可学习，更新后Query条件梯度生效，原decoder残差路径仍正常训练。

每层OST-DA共272,932参数，原attention为74,016参数；DEIM-M四层合计新增795,664参数。
每层都执行新投影和门控，推理保留eval_idx之前所需的所有交叉注意力层。
显存、吞吐及检测收益需服务器实测，不能由拓扑结构预先保证抑制廊桥或提升类别区分。

## COCO初始化和采样分配对照

原baseline每头三个尺度的点数是[3,6,3]，附件要求[4,4,4]。总点数仍为12，但分层分配不同。
提供独立uniform4对照；评价OST-DA自身收益应与同点数分配的对照比较。
三方法组合也有对应uniform4对照，避免将重新分配采样点的影响混入注意力收益。

沿用 `-t` 的同名同形加载：原offset/attention参数可加载COCO权重；新增angle、value、gate、output
参数使用新初始化。COCO加载可能覆盖offset/attention的构造期初始化，与uniform4对照采用同一规则。
旧attention的num_points_scale缓冲区不再需要；原加载器按当前模型键和形状筛选，不改变旧配置加载逻辑。
恢复OST-DA运行时使用其完整checkpoint和保存的配置 `-r`，不要用旧baseline optimizer状态恢复新结构。

## 文件、配置与固定协议

- `engine/deim/orthogonal_topology_attention.py`：几何、value适配、投影、采样和门控。
- `engine/deim/dfine_decoder.py`、`engine/deim/__init__.py`：替换开关及类导出。
- `tools/run_sar_module_experiment.sh`：记录角度/头/点分配、扰动范围、投影和AMP设置。
- `tests/test_ost_da.py`：解析几何、手算插值、梯度、checkpoint及真实训练路径。
- `deim_hgnetv2_m_sar_aircraft_uniform4_58e_b28_old_server.yml`：均匀4点baseline对照。
- `deim_hgnetv2_m_sar_aircraft_ost_da_58e_b28_old_server.yml`：OST-DA单方法。
- `deim_hgnetv2_m_sar_aircraft_spi_module_wfdr_uniform4_58e_b28_old_server.yml`：SPI+W-FDR均匀4点对照。
- `deim_hgnetv2_m_sar_aircraft_spi_module_wfdr_ost_da_58e_b28_old_server.yml`：三个方法组合。

四配置均在 configs/deim_dfine/ 下，有独立output_dir。结构常数为对半分头/分点、扰动界0.25，
不添加额外损失或新的训练调度。其余沿用同58轮协议：DEIM-D-FINE-M / HGNetv2-B2、7类ID0–6、
train3047 / val442 / test879、seed0、GPU0/1、全局batch28（每卡14）、基础/验证640、训练多尺度480–800，
warmup432、flat29、no_aug8、lr_gamma0.5，增强policy [4,29,50]、mixup [4,29]、stop50，
AdamW主LR3.5e-4、骨干非BN LR3.5e-5、WD1e-4、EMA、AMP FP16、FP32 loss/retry及展平top300后处理。
组合配置继承SPI和W-FDR的既定结构/权重。所有对照和方法均从同一COCO M权重 `-t` 新训。
数据沿用现有SAR-AIRcraft目录及0–6标注，无需重新准备。
启动器归档配置、Git revision、seed、batch、输入、优化器、增强、EMA/AMP、初始化和评估协议。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_ost_da_58e_b28_old_server.yml
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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_spi_module_wfdr_ost_da_58e_b28_old_server.yml
```

## 日志、指标、TensorBoard及恢复

重连后显式设置OUT为启动器打印的目录并激活环境：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际OST-DA运行目录
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
    losses = {key:value for key,value in row.items() if key.startswith('train_spi_module_') or key in ('train_loss_mal', 'train_loss_bbox', 'train_loss_fgl') or key.startswith('train_loss_wfdr')}
    print(label, 'epoch=', row['epoch'],
          'AP/AP50/AP75=', [round(value * 100, 4) for value in row['test_coco_eval_bbox'][:3]],
          'losses=', losses)
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

JSONL指标在log.txt；COCO bbox第0/1/2项是AP/AP50/AP75，百分数报告乘100。
TensorBoard在summary/，训练COCO数据在eval/latest.pth。
原进程停止后恢复同一OST-DA实验：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原OST-DA运行目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复用resolved_config.yml和 `-r last.pth`，不同时传 `-t`；日志为打印的resume_*.console.log。
本协议last.pth正常最后更新于epoch49，不是最后一轮权重。

## 按验证AP选择权重并评估测试集

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原OST-DA运行目录
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

## 运行相同采样分配的对照

前一运行结束、两卡空闲后，在相同环境分别启动对照，仍由启动器创建新目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
CONFIG=configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_uniform4_58e_b28_old_server.yml
# 三方法组合的对照改用：
# CONFIG=configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_spi_module_wfdr_uniform4_58e_b28_old_server.yml
bash tools/run_sar_module_experiment.sh "$CONFIG"
```

## 本地功能验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_ost_da.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_spi_module.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_wfdr.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sdcp.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_clcr_lite.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_clcr.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_query_prototypes.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_pgd_rem.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sar_58e_protocol.py -v
```

验证独立角度/自由头/双轴偏移、扰动界、共享及分层参考框、非方形图的手算双线性插值、
零填充、原value拆分适配、投影与门控公式、FP32/CPU BF16 AMP梯度、原权重/RNG、COCO同形加载、
aux宽层、普通/aux/DN训练、SPI/W-FDR组合、恢复及部署，并验证4个配置的采样预算及协议。
结果：OST-DA 12项、SPI-Module 12项、W-FDR 12项、SDCP 14项、Lite 11项、原CLCR 10项、
原decoder/MPQ回归7项、PGD-REM 6项、58轮协议5项，合计89项全部通过。
启动器、本文7段Bash命令及嵌入Python片段语法检查通过，没有执行服务器命令。
本地不包含真实数据训练和CUDA双卡吞吐/显存测试，不据此宣称检测增益。
