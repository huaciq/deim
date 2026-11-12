# E013 方向共享编码与方向证据融合

> 本文保留 E013-v1 的历史设计。2026-11-12 复核发现：检查点开关同时改变 BN 批次组织，且重计算重复更新滑动统计，
> 因此原“梯度不变”的纯卷积测试不能外推到真实编码器。D18 的亮度统计及计算量对照也有更正。
> 正式推进的独立配置改为 [E013-v2：投影后融合、单次编码器主体](sardet100k_orientation_fusion_v2.md)，不修改本版实验设置。

日期：2026-11-12。只针对 SARDet-100K。状态：已实现并通过定向测试，**尚未训练**；训练前检验（[D18](sardet100k_d18_orientation_cue.md)）不支持它，预期与基线持平，见[训练前检验](#训练前检验d18)。代码 `engine/deim/orientation_fusion.py`，接入点 `engine/deim/deim.py`（默认关闭），配置 `configs/deim_dfine/deim_hgnetv2_m_sardet100k_orientation_fusion_36e_b32.yml`，测试 `tests/test_orientation_fusion.py`。

## 问题

来自 [D17](sardet100k_d16_d17_source_orientation.md)：检测器的目标证据是按图像方向分开学的。

- 冻结权重，把 val 图垂直翻转，AP 从 65.3 掉到 38.2，转置掉到 25.1；水平翻转（训练增强里有）不变。没有副本的目标同样从 44.5 掉到 24.8 和 17.4。E008 相同。
- 训练集里只以另一个方向出现过的目标，检出率不高于从未出现过的目标（OGSOD bridge：72.7% 对 77.8%，同方向见过的是 93.4%）。
- 原图漏检的无副本 aircraft 和 bridge，有 15–38% 在垂直翻转后能检出，是水平翻转对照的两倍。

## 结构

```
图像 x ──┬──────────────► 共享 HGNetv2 ─► 共享 HybridEncoder ─► F0（三层）──────────┐
         │                    （同一组权重，一次前向）                               ├─► 方向证据融合 ─► Decoder（不变）
         └─ 垂直翻转并平移 ──►        同上                    ─► 翻回并对齐 ─► F1 ──┘
```

1. **方向展开。** 输入图加上它的 K−1 个方向视图。E013 取 K = 2：原图和垂直翻转。水平翻转仍由数据增强提供，两者合起来覆盖翻转的 4 个方向。模块也支持转置和 90° 旋转视图（需要正方形输入）。
2. **共享编码。** 所有视图拼成一个批次，过同一套 backbone 和 encoder。没有新增的编码参数，COCO 初始化全部保留。
3. **对齐。** 把视图的三层特征翻回原方向，并落到原图的网格上（见下节）。
4. **方向证据融合。** 每一层、每个位置，把 K 路对齐后的特征拼起来过一个门控（1×1 卷积 → BN → ReLU → 1×1 卷积），给每个通道组（8 组）输出一组对 K 路的 softmax 权重，加权求和得到融合特征。
5. Decoder、匹配、损失、增强（含 Dense O2O）、日程都是基线 E000 的。

融合前各方向的特征各自保留，所以如果数据里确实存在稳定的成像方向线索，模型仍可以用原图那一路；改变的是每个方向的样本都在训练同一组卷积核。

**初始化：** 门控最后一层权重为 0，原图那一路的偏置为 2，所以开始时权重与内容无关，原图占 88%，Decoder 起步时看到的接近它原本的输入。

**新增参数：** 三层门控共 101,808 个（19.48M → 19.58M）。**计算量：** backbone 和 encoder 约为 2 倍，Decoder 不变。

## 显存与双卡

协议不变：GPU 4、5 双卡，全局批量 32（每卡 16 张图）。在 GPU 4 上用合成数据测了一步训练（前向、损失、反向，AMP，16 张图）的峰值显存：

| 实现 | 640 尺度 | 800 尺度（多尺度训练的最大尺寸） |
|---|---|---|
| 基线 E000 | 10.6 GB | 15.0 GB |
| E013，两个视图一次编码 | 17.5 GB | 超出 24 GB |
| E013，只对翻转视图做梯度检查点 | 14.3 GB | 22.5 GB |
| E013，每个视图都做梯度检查点（配置默认） | 7.6 GB | 11.9 GB |

基线真实训练日志里的峰值是 18.2 GB，比合成步高约 3 GB，所以只有最后一种在双卡上有余量。梯度检查点不保存各视图的中间结果，反向时逐个视图重算，输出和梯度不变（测试已核对），代价是每个视图多一次编码前向，训练时间估计约为基线的 2 倍（约 36–40 小时）。

两点没有验证：这个估计的训练时间；梯度检查点与 SyncBN 在多卡下同时使用（BN 的滑动统计每步会更新两次）。如果改用 4 张卡，可把 `checkpoint_views` 设为 False。

## 网格对齐

偶数尺寸上的步长 2 卷积和 HGNetv2 的 stem 采样不对称。翻转图像的特征翻回来以后，与原图特征差将近一格。做法是：翻转后的图像先平移 3 个像素再编码，翻回的特征再移 1 格。

标定方法：把卷积核改成翻转对称的，位置编码置零，图像内容只放在中心，这样剩下的差异只来自网格错位。

| 位置 | 不补偿（P3 / P4 / P5 相对误差） | 平移 3 像素、移 1 格 |
|---|---|---|
| backbone 输出 | 36% / 68% / 85% | 1.0% / 0.7% / 0.5% |
| encoder 输出 | 57% / 82% / 86% | 45% / 61% / 0.6% |

backbone 三层和 encoder 的 P5 可以对齐到 1% 以内。encoder 的 P3、P4 对不齐，原因是自顶向下路径里的最近邻上采样本身带半格偏置，在翻转视图里这个偏置是镜像的，任何整体平移都消不掉。这部分保持原样：每一路特征在自己的坐标系里是自洽的，网格中心已经对齐，差别只在高层语义注入的位置偏了半格。这是本结构已知的不精确之处。

## 已做的验证

`python -m unittest tests.test_orientation_fusion -v`，CPU，9 项通过：

- 配置相对基线 E000 只多出 `DEIM.orientation_fusion` 一块。
- 7 种视图在不加平移时，变换后再对齐能精确还原。
- 初始门控与内容无关，原图权重为 e²/(e²+1)。
- 所有视图只经过一次编码，编码器的参数从融合结果得到梯度。
- 基线的全部参数能按相同形状载入，新增参数只有门控。
- 把门控固定在原图一路时，整个检测器的输出与基线相同（接线正确）。
- 开关梯度检查点，输出和梯度相同。
- 默认平移量在真实 backbone 上使翻转视图与原图对齐到 5% 以内，不补偿时误差超过 30%。

没有做的：没有跑过带真实数据的训练步；没有在多卡 DDP 下运行过。

## 训练前检验（D18）

用户要求在训练前先核实“上下不对称是不是有用的线索”。结果见 [D18](sardet100k_d18_orientation_cue.md)，不使用检测器，只用目标裁块：

- 上下不对称是目标自身的性质，train 与 val 一致；只看一个裁块就能以 78% 的准确率认出它有没有被翻转，aircraft 和 car 超过 91%，强杂波只有 58%。
- 用增强强行共享方向，“目标对强杂波”的召回率在原方向上下降 2.5–3.7 点（8 个方向下降 4.1–5.9 点），超过事先定的 2 点界线。
- E013 的缩小版（同一主干看原图和翻转图、门控融合）是 73.1，单视图 73.9，差值在种子波动内；同样的计算用在加宽上是 76.6。

结论：E013 避免了增强的损失，但没有显示收益，预期与基线持平。D17 里“方向覆盖不足限制泛化”的假设在裁块层面没有得到支持。

## 判定（训练前写定）

对照是基线 E000（val AP 65.29，没有副本的目标 AP 44.5，垂直翻转后 AP 38.2）。只看 val。

| 项 | 标准 |
|---|---|
| 主指标 | val AP 相对 E000 提高 ≥ 0.5 |
| 机制 1 | 没有副本的目标 AP 高于 44.5；512 aircraft 和 OGSOD bridge 的无副本召回率上升 |
| 机制 2 | 门控给翻转一路的平均权重明显高于初始的 12%，说明另一路被用上 |
| 记录 | 垂直翻转 val 图后的 AP（E000 是 38.2） |
| 证伪 | 主指标未达到，即放弃这条线，不调门控、不换视图组合 |

垂直翻转后的 AP 不再作为机制指标：D18 说明模型有理由保留方向性，它不追平原图不代表结构失败。机制评估用 `tools/sardet100k/orientation_robustness.py`，它直接调用整个模型，不需要改动。

只加方向增强的对照 E012 不再需要：D18 预示它会下降，而 E013 的判定只看它是否超过 E000。

## 服务器命令（8 卡服务器，GPU 4、5）

更新代码：

```bash
cd /root/zxy/deim
conda activate deim_zxy
git status --short --branch
git fetch origin
git switch main
git -c pull.rebase=false pull --ff-only origin main
git rev-parse --short HEAD
export CFG=configs/deim_dfine/deim_hgnetv2_m_sardet100k_orientation_fusion_36e_b32.yml
python -m unittest tests.test_orientation_fusion -v
```

启动（launcher 新建目录，nohup 后保存 PID 并 disown）：

```bash
GPU_IDS=4,5 EXPERIMENT_ROOT=/root/zxy/experiments \
INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth \
RUN_PREFIX=orientfusion \
bash tools/run_sar_module_experiment.sh "$CFG"
```

把启动输出的目录设为 OUT。重连后要重新激活环境并设置 OUT。

```bash
conda activate deim_zxy
export OUT=/root/zxy/experiments/orientfusion_deim_hgnetv2_m_sardet100k_orientation_fusion_36e_b32_seed0_<时间戳>_<后缀>
tail -f "$OUT/console.log"
nvidia-smi -i 4,5
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
tail -n 5 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 127.0.0.1 --port 6006
```

中断后恢复（先确认原进程已停止）：

```bash
GPU_IDS=4,5 bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

查看逐轮指标并按 val AP 选权重：

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['OUT'])
rows = [json.loads(x) for x in (root/'log.txt').read_text().splitlines() if x.strip()]
for row in rows[-5:]:
    a = row['test_coco_eval_bbox']
    print(row['epoch']+1, dict(zip(['AP','AP50','AP75','APs','APm','APl'], [round(100*v, 2) for v in a[:6]])))
best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
print('best', best['epoch']+1, round(100*best['test_coco_eval_bbox'][0], 2), 'best_stg2' if best['epoch'] >= 32 else 'best_stg1')
PY
```

`last.pth` 只在无增强阶段之前更新，不是最后一轮。COCO bbox 数组的 0、1、2 是 AP、AP50、AP75，乘 100 报百分数。逐轮指标在 `log.txt`，TensorBoard 在 `summary/`。

训练结束后的机制评估（val，8 个方向）：

```bash
R0=/root/zxy/experiments/deim_m_sardet100k_deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_seed0_20261105_125147_N2zAJh
D=$R0/independent_verify_20261106/d17_orientation
cd /root/zxy/deim && export PYTHONPATH=/root/zxy/deim
CUDA_VISIBLE_DEVICES=0 python $D/code/orientation_robustness.py predict "$OUT" $D --name E013
python $D/code/orientation_robustness.py ap $D --d12 $R0/independent_verify_20261106/d12_generalisation --names E013
```

## 局限和风险

- **训练前检验不支持。** 上下不对称是真实、稳定、有用的线索（D18），结构的缩小版没有收益。
- **不针对小 tank。** D17 里小 tank 对方向不敏感。
- **对齐不完全。** encoder 的 P3、P4 在自顶向下路径上有镜像的半格偏置（见上）。
- **官方 val 里换了方向的副本只有 1%**，增益必须来自没有副本的目标。
- **计算量约翻倍**，与基线的速度比较需要单独报告。
- 等变和多方向共享在遥感检测里有先例（如 ReDet），这里的做法是共享预训练编码器加可学习的方向融合，不是严格的等变网络。
