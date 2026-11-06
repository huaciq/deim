# M1：成像—标注几何解耦（SARDet-100K）

代码：`engine/deim/imaging_geometry.py`，配置：`configs/deim_dfine/deim_hgnetv2_m_sardet100k_m1_geometry_36e_b32.yml`。
证据和其他候选见[基线诊断](sardet100k_baseline_diagnosis.md)，结果记入[实验总表](sardet100k_results_ledger.md)。

## 问题

bridge（97%）、harbor（100%）和 60% 的 tank GT 来自 OGSOD，这个来源的 AP 只有 52.04，MSAR 是 84.30。两个几何问题直接压低了官方 AP：

1. OGSOD 的预测框相对标注有约 +0.8 原图像素的 x 向偏移。翻转增强让模型只能定在响应中心。
2. 13% 的 aircraft 官方框超出图像，训练目标却被裁到图内。

## 结构诊断决定了模型的形式

全部来自 E000 的 val 预测，结果在运行目录 `independent_verify_20261106/geometry/report.json`。

| 问题 | 结果 | 对模型的约束 |
| --- | --- | --- |
| 四条边怎么偏 | OGSOD bridge：左 +0.81，右 +0.85，上 +0.04，下 +0.02；tank 同样 | 是平移，不是逐边残差 |
| 随大小、位置变化吗 | bridge dx 随宽度三分位 0.85、0.82、0.84；斜率 0.002 px/px；随 x 位置基本不变 | 是常量，不需要逐目标回归 |
| 单张图的偏移能预测吗 | 图像 CNN 交叉验证 R² −0.10；同图留一 R² 0.04；类别、尺寸、位置的 R² 都低于 0.02 | 不做逐图自由回归，只用少数离散的域 |
| 其余方差是什么 | dx 方差 1.79，均值平方只占二阶矩 27% | 其余基本是噪声，要用稳健损失 |
| aircraft 偏差的结构 | dx 随 x 位置 +0.70、−0.61、−1.23；dy 随 y 位置 −0.05、−0.69、−1.37 | 和截断一致，只作用于贴边的边 |
| 截断框能否推断 | 对可见部分 IoU 0.909，对官方框 0.771；同图尺寸一致只到 0.808；超出中位数 10 像素（长边的 12.8%） | 需要专门的分支，并从目标外观推断 |

这就是这里不用更深回归网络的原因：数据显示偏移本身很简单，单图的变化又无法从图像预测，更大的自由度只会拟合噪声。

## 模型

    官方框 = 响应框 + 几何域平移 + 截断补全

**响应框。** 原 D-FINE 路径。训练时匹配、去噪、FDR、DDF、MAL 的目标全部换成响应框：官方框减去平移，再裁到图内。模型结构不变，预测随翻转等变。

**几何域平移。** 每张图一个二维向量：

    τ = Σ_k π_k(g) · t_k

- `t_k`：K=4 个可学习的域平移，零初始化，所以起点就是基线。
- `π(g)`：由 encoder 各层全局均值和 5 个对数强度统计量（均值、方差、三个分位数，严格翻转不变）算出。特征已停止梯度。
- 向量定义在未翻转的成像方向下。图像被翻转时，`τx` 变号。这是保留翻转增强又能学到偏移的关键。
- 限制成少数离散的域，是因为单图偏移无法预测。

**截断补全。** 只作用于贴边的边（到边界距离 < 0.01）：

    e_j = softplus(MLP(query, 框内特征, 框周特征, 框, 边界距离)) · gate_j

- 输出非负，单位是框的边长。不贴边的边恒为 0。
- 特征停止梯度。最后一层零权重、偏置 −8，所以起点约等于不动。
- 在当前图像坐标下工作，翻转时不需要变号。

**推理。** 响应框 → 贴边补全 → 平移。结构上只多这个头。

## 损失

只用最后一层匹配上的目标（与响应目标 IoU ≥ 0.5）和干净样本。Mosaic 和 MixUp 样本混合多张图，不属于单一几何域，保持原目标，也不参与这两项损失。

    L_translation = SmoothL1(τ, (官方中心 − 预测中心) 转到未翻转方向), beta = 0.5 px
    L_extension   = L1(e, 官方溢出 / 预测框边长), 只算贴边的边

截断或贴边的目标不参与平移估计。两项都只回传到几何头，权重均为 1，单位是 1/256 图像宽度。

## 数据流

- `CocoDetection(keep_overflow=True)` 在裁剪前记录四条边的溢出比例，并打包进 `labels`。torchvision 不允许同一样本有两个 BoundingBoxes，打包后才能和目标一起经过 Mosaic 和 Sanitize。
- `GeometryHorizontalFlip` 替换 `RandomHorizontalFlip`，记录翻转状态并交换左右溢出。
- `GeometryTargets` 放在管线末尾，拆回 `labels` 和 `official_overflow`。
- 评估管线、评估代码和官方标注都不变。

## 判定（预先写定）

只看 val，test 只在方法定稿后评估一次。

| 项目 | 内容 |
| --- | --- |
| 协议 | 与 E000 相同（36 轮，b32，seed 0，新服务器），差异只有上述几何设置 |
| 主指标 | val AP 比 E000（65.29）涨 1.0 以上；单种子，要标明 |
| 机制指标 1 | OGSOD val 的 dx 绝对值降到 0.2 以下，不靠来源信息 |
| 机制指标 2 | 超出图像的 aircraft GT 的 TP IoU 中位数高于 0.771 |
| 机制指标 3 | 学到的 `t_k` 中有一个接近 (+0.8, 0)，且 OGSOD 图主要分配给它 |
| 证伪 | dx 不降，或 AP 涨幅不到 0.5 |
| 消融（有信号后） | 只开平移；只开补全；两者都开；只关闭翻转（G1）作对照 |

## 局限

- 模型学的是官方标注的几何，不代表物理定位更准。论文要讲“分离雷达响应和标注几何”，不能写成检测更准。
- 偏移的物理原因未证实，OGSOD-1.0 的标注流程还没读到原文。
- 增益上限只有零训练估计：常量平移 +1.85，不看图的头 +1.04，裁到图内后 aircraft 估计 +3.98。三者有重叠，不能相加。
- 对比方法也有同样的偏差，论文比较时要说明。
- 文献里有没有类似的解耦还没查。

## 功能验证

```bash
CUDA_VISIBLE_DEVICES= python -m unittest tests.test_imaging_geometry -v
```

覆盖：溢出记录和翻转交换；Sanitize 后对齐；Mosaic 和 MixUp 不算干净样本；平移符号；贴边门控；带噪合成数据上能恢复 +0.8 和 0 两个域；几何损失不回传到检测器；配置除几何设置外与 E000 相同。有服务器数据时还会走真实数据管线。

## 服务器命令（新服务器）

服务器访问 GitHub 超时时，用 Git bundle 传提交，再执行 `git merge --ff-only FETCH_HEAD`。

更新代码：

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
```

启动（两张卡都空闲才会启动）：

```bash
EXPERIMENT_ROOT=/root/zxy/experiments INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth RUN_PREFIX=deim_m_sardet100k bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sardet100k_m1_geometry_36e_b32.yml
```

重连：

```bash
export OUT=$(ls -dt /root/zxy/experiments/deim_m_sardet100k_*m1_geometry* | head -n 1); echo "$OUT"
tail -f "$OUT/console.log"
```

GPU：

```bash
watch -n 2 nvidia-smi
```

指标：

```bash
python -c 'import json,os; rows=[json.loads(s) for s in open(os.environ["OUT"]+"/log.txt") if s.strip()]; r=max(rows,key=lambda x:x["test_coco_eval_bbox"][0]); print(r["epoch"],[round(100*x,2) for x in r["test_coco_eval_bbox"][:6]])'
```

几何损失：

```bash
grep -E "Epoch: \[[0-9]+\]  \[[0-9]+00/" "$OUT/console.log" | tail -n 3 | grep -oE "loss_geo_[a-z]+: [0-9.]+ \([0-9.]+\)"
```

学到的域平移：

```bash
python -c 'import os,torch; s=torch.load(os.environ["OUT"]+"/last.pth",map_location="cpu")["ema"]["module"]; print(s["decoder.imaging_geometry.domain_translations"])'
```

TensorBoard：

```bash
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

恢复（确认进程已停止）：

```bash
cd /root/zxy/deim && bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

## 结果

待训练完成后填写。
