# SAR-AIRcraft OST-DA v2 实现与实验交接

## 本次改动及复现边界

本版本实现三项修改：各尺度采样数 [4,6,4]、拓扑点强制正负镜像、零初始化残差门控。
新实现为 `engine/deim/symmetric_topology_attention.py` 的
`SymmetricOrthogonalTopologyDeformAttn`，替换 Decoder 每层的 cross_attn。
使用 `DFINETransformer.use_ost_da: True` 和 `ost_da_version: v2` 显式启用。
未指定版本时仍为 v1：旧类、旧配置、旧 checkpoint 的计算路径保持原样，
已报告的 [4,4,4] 实验可继续复现和恢复，不在原结果背后更换算法。

### 1. 恢复 stride-16 的采样密度

`num_points: [4,6,4]` 对应 stride-8/16/32，每头共14点。
也支持标量6，即 [6,6,6]；v2 每个尺度必须为不小于4的偶数。
每层按 n_points_list 切分，坐标按当前尺度 K_l 归一化：

`location_l = ref_xy_l + offset_l / K_l * ref_wh_l * 0.5`。

位置存储为 [B,Q,H,14,2]，注意力为 [B,Q,H,14]，在每头全部14点上 softmax。
复用原生 `deformable_attention_core_func_v2` 的可变点数 bilinear 路径，
按 [4,6,4] 分别对三张特征图采样，无填充点或重复计算的虚拟点。
仍支持框 [B,Q,1,4] 广播和 [B,Q,3,4] 分尺度参考框，以及原生分层/扁平 value。
采样不进行框内截断，zeros padding、align_corners=False，几何与采样核为 FP32。

### 2. 强制十字对称采样

8头仍分为4个自由二维头和4个拓扑头。自由头不加对称约束。
每个 Query 的每个拓扑头仍独立预测 theta=pi*sigmoid(angle_proj(query))。
两条局部坐标轴为 u_body=(cos(theta),sin(theta))、u_wing=(-sin(theta),cos(theta))。
拓扑头只预测 K_l/2 个标量步长，没有独立负半轴输出，也不再预测侧向扰动。
每个步长 r 生成一对 [r*u, -r*u]，点的顺序为机身正负点对在前、机翼正负点对在后。

- K=4：每轴一个步长，点位为 [r_f*u_body,-r_f*u_body,r_w*u_wing,-r_w*u_wing]。
- K=6：3个正负点对不能在两轴均分，拓扑头0/2采用机身4点+机翼2点，
  拓扑头1/3采用机身2点+机翼4点；四个拓扑头合计的两轴采样密度相同。
- K=8：两轴各两个点对。其余偶数同理，额外点对在相邻头之间交替分配。

步长投影权重为0，bias在每条轴上设为1,2,...；angle权重/bias为0，初始 theta=pi/2。
因此初始拓扑点为非退化的中心对称十字，而不是从极坐标 bias 取分量形成单侧点。
步长更新后可取任意实数，正负配对始终成立；不会用 abs/截断导致额外的不可微约束。
每对共享同一步长，两个点的坐标梯度也正负耦合。注意力权重仍独立学习，不强制配对等权。
正交性定义在框尺度归一化坐标；乘非正方形框的宽高后，像素坐标夹角不保证为90度。

### 3. 零初始化残差门控

不使用 v1 的 Linear -> LayerNorm -> Sigmoid 乘法衰减，改为：

```python
self.topo_gate_linear = nn.Linear(d_model, d_model)
self.gate_alpha = nn.Parameter(torch.zeros(1, 1, d_model))
gate = torch.tanh(self.topo_gate_linear(output))
output = self.output_proj(output * (1.0 + self.gate_alpha * gate))
```

只将 alpha 置零，门控 Linear 正常初始化，避免双零初始化让残差分支无法学习。
初始时门控乘数严格为1，等价于 output_proj(aggregated)，不衰减聚合信号。
首步 alpha 有梯度、gate Linear 梯度为0；alpha更新后，门控权重也开始学习。
这只保证新增门控初始不衰减，不表示整层交叉注意力与原 baseline 等价。
value_proj/output_proj、Decoder 原 gateway、自注意力、FFN、FDR、LQE均保留。

### COCO初始化、参数和比较口径

自由偏移使用独立的 free_sampling_offsets 名字，轴向使用 topo_steps，
防止 COCO 的非对称二维 offset 权重覆盖对称初始化。
当前 COCO 原分配为 [3,6,3]，attention_weights 的形状与14点版本不同，也保留新初始化。
其余原模型的同名同形参数沿用原来的 -t 加载规则。
模块在 fork_rng 中替换；与 [4,6,4] 原注意力对照相比，共同参数和后续随机状态一致。

D=256/H=8/L=3/[4,6,4] 时，v2 每层交叉注意力263,424参数；
原 [3,6,3] attention 为74,016参数，4层合计新增757,632参数。
相较旧 OST-DA，4层少38,032参数。最终推理仍执行这些投影、采样和门控。
点数从12增至14会增加采样成本，尚未实测CUDA吞吐及显存。

新增两份独立配置，均位于 configs/deim_dfine/：

- deim_hgnetv2_m_sar_aircraft_sampling464_58e_b28_old_server.yml：仅改原attention的采样分配，对照组。
- deim_hgnetv2_m_sar_aircraft_ost_da_v2_58e_b28_old_server.yml：启用完整v2。

已有 OST-DA 相对旧baseline的验证集最佳AP75约+0.85 AP，不能单凭此证明全部机制或新版增益。
本次修改的目标是改善初始化和早期优化；是否消除前25轮慢热、是否保留AP75收益，需新训曲线验证。
与 [4,6,4] 对照比较才能区分点数增加和拓扑/门控改动带来的影响。

## 固定训练协议

DEIM-D-FINE-M / HGNetv2-B2、7类ID0–6，train3047 / val442 / test879；
seed0、58轮、GPU0/1、全局batch28（每卡14）、基础/验证640、训练多尺度480–800；
warmup432、flat29、no_aug8、lr_gamma0.5；AdamW主LR3.5e-4、骨干非BN LR3.5e-5、WD1e-4；
EMA、SyncBN、AMP FP16和FP32 loss/retry；增强policy [4,29,50]、mixup [4,29]、stop50。
RandomZoomOut和RandomIoUCrop仍全程关闭；Mosaic、MixUp、光度扰动、翻转及多尺度沿用原协议。
从同一COCO M权重 -t 新训，未将SPI或W-FDR默认为启用，也不修改原数据和标注。
数据沿用 /home/zxy/sar/datasets/SAR-AIRcraft-1.0/COCO/{train,val,test}.json，无需重新准备。
启动器记录配置、Git revision、seed、batch、输入、LR、增强、EMA/AMP、初始化和评估协议，
并额外记录v1/v2、半轴步长和残差门控规则。以下输出根目录沿用当前实验使用的 /data2 路径。

## 服务器 Git 更新和双卡新训

工作树若有改动或分叉，先检查并保留，不使用强制覆盖。然后执行：

```bash
cd /home/zxy/sar/repos/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/data2/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_ost_da_v2_58e_b28_old_server.yml
```

启动器创建独立新目录，实际命令为 CUDA_VISIBLE_DEVICES=0,1 与
torchrun --standalone --nproc_per_node=2 train.py，COCO初始化使用 -t。
以 nohup 后台执行，重定向 console.log 2>&1 < /dev/null，先保存 $! 至 launcher.pid 再 disown。
在两卡空闲后，如需相同采样密度的对照，独立启动：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/data2/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_sampling464_58e_b28_old_server.yml
```

## 查看进程、日志和指标

将OUT替换为启动器打印的实际目录；每次重连都重新激活环境、设置OUT：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为实际OST-DA-v2目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C退出跟踪，再读取峰值、最终轮及前25轮最好指标：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(s) for s in (Path(sys.argv[1]) / 'log.txt').read_text().splitlines() if s.startswith('{')]
by_epoch = {row['epoch']: row for row in rows if 'test_coco_eval_bbox' in row}
rows = [by_epoch[epoch] for epoch in sorted(by_epoch)]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
best75 = max(rows, key=lambda row: row['test_coco_eval_bbox'][2])
early = [row for row in rows if row['epoch'] < 25]
selected = [('best_AP', best), ('best_AP75', best75), ('last', rows[-1])]
if early:
    selected.append(('first_25_best_AP', max(early, key=lambda row: row['test_coco_eval_bbox'][0])))
for label, row in selected:
    print(label, 'epoch=', row['epoch'], 'AP/AP50/AP75=',
          [round(v * 100, 4) for v in row['test_coco_eval_bbox'][:3]])
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

log.txt为JSONL，bbox指标0/1/2分别是AP@[.50:.95]/AP50/AP75，报告百分数时乘100。
summary/为TensorBoard，训练评估数据为eval/latest.pth。

## 恢复同一v2运行

原进程停止后使用保存的resolved_config.yml和 -r last.pth，恢复时不传 -t：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为原OST-DA-v2目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

启动器打印resume_*.console.log路径，并保存恢复PID/命令和元信息。
当前solver的last.pth正常最后更新于epoch49，不代表最终第57轮权重。
旧v1运行继续使用其原保存配置恢复；新v2实验从COCO初始化开始。

## 按验证AP选权重并评估测试集

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为原OST-DA-v2目录
CKPT=$(python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
import yaml

run = Path(sys.argv[1])
cfg = yaml.safe_load((run / 'resolved_config.yml').read_text())
by_epoch = {r['epoch']: r for s in (run / 'log.txt').read_text().splitlines()
            if s.startswith('{') for r in [json.loads(s)] if 'test_coco_eval_bbox' in r}
best = max(by_epoch.values(), key=lambda row: row['test_coco_eval_bbox'][0])
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

按实际验证AP选best_stg1.pth/best_stg2.pth，独立评估数据在 $EVAL/eval.pth，指标见控制台。

## 本地功能验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_ost_da_v2.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_ost_da.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sar_58e_protocol.py -v
```

验证范围：可变点数、初始及学习后的镜像坐标/梯度、6点两轴密度平衡、零门控的精确公式和两步学习、
非方形特征图手算双线性插值/零填充、原生value适配、FP32及CPU BF16 AMP、COCO同形加载、
共同参数/RNG、aux宽层、普通/DN/aux训练及SPI+W-FDR兼容、checkpoint/部署、v1回归和58轮协议。
本地没有执行真实数据58轮训练、CUDA双卡吞吐或显存测试，不据此宣称消除慢热或提高AP。

验证结果：v2新增12项、v1回归12项、58轮协议5项，共29项全部通过。
启动器及本文6段Bash命令、其中和启动器中的5段嵌入Python语法检查通过；未执行服务器命令。
参数统计实测为v2每层263,424，密度对照每层86,352，原baseline每层74,016。
