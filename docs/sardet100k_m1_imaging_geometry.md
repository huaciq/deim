# M1：成像—标注几何解耦（SARDet-100K）

> 2026-11-06：用户决定 M1 不作为研究方法，因为它修正的是数据集标注，所有方法都面临同样的问题。这里只保留记录，主线见 [GREA 设计](sardet100k_grea_design.md)。

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

## 与 RSAR 重标注的对比（2026-11-06，val）

RSAR 由 SARDet-100K 重标注而来，文件名相同。把它的旋转框转成外接水平框，与官方框和 E000 预测（score≥0.3）比较中心偏移，单位是原图像素，均值 ± 标准误。脚本 `tools/sardet100k/rsar_offset_check.py`，结果在运行目录 `independent_verify_20261106/rsar/report.json`。

| 来源 / 类别 | n | RSAR − 官方 dx | RSAR − 官方 dy | 预测 − 官方 dx | 预测 − RSAR dx | 预测 − RSAR dy | 框完全相同 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OGSOD bridge | 2,787 | +0.50 ± 0.04 | +0.47 ± 0.04 | +0.80 | +0.19 | −0.46 | 0% |
| OGSOD harbor | 371 | +1.01 ± 0.17 | +0.25 ± 0.18 | +0.97 | −0.30 | −0.33 | 0% |
| OGSOD tank | 773 | 0 | 0 | +0.93 | +0.93 | +0.05 | 100% |
| other ship | 8,366 | +0.85 ± 0.02 | +1.02 ± 0.02 | +0.02 | −0.85 | −1.07 | 0.2% |
| other car | 1,216 | +0.51 ± 0.06 | −1.47 ± 0.06 | +0.26 | −0.24 | +1.59 | 0% |
| MSAR ship | 1,998 | −0.18 ± 0.04 | −0.05 ± 0.04 | +0.01 | +0.21 | +0.06 | 0.1% |

判读：

1. **RSAR 不能当作中性真值。** tank 的框与官方完全相同，无法检验。bridge、harbor 的 dx 确实向预测靠近，但 dy 多出了偏差，ship 和 car 上 RSAR 还有预测里没有的系统偏移。
2. **“响应框就是真实位置”没有得到支持。** 论文不能这样写。
3. **确认了标注惯例依赖来源和类别。** 同一批图的两套公开标注，差异最大约 1.5 像素，方向随来源和类别变化。
4. **局限：** 外接框会放大倾斜目标，所以 IoU 不直接比较，只看中心。"other" 没有细分来源。RSAR 只覆盖 8,467 张 val 图。

### 偏移来自标注、模型还是定义差异（2026-11-06，train 和 val）

脚本 `tools/sardet100k/ogsod_offset_evidence.py`，结果在运行目录 `independent_verify_20261106/offset_evidence/`。只抽样 train（25,000 张）和 val，未读 test。

预测 − 官方（score≥0.3，IoU≥0.5，区间按图 bootstrap）：

| 来源 / 类别 | n | 图数 | dx | 95% CI | dy | dx>0 占比 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| OGSOD bridge | 2,442 | 1,396 | +0.78 | 0.73–0.83 | +0.03 | 78% |
| OGSOD tank | 943 | 79 | +0.73 | 0.59–0.87 | +0.05 | 82% |
| OGSOD harbor | 131 | 49 | +0.78 | 0.51–1.06 | −0.06 | 79% |
| MSAR tank | 419 | 54 | +0.03 | 0.00–0.06 | +0.02 | 52% |
| MSAR ship | 1,445 | 927 | +0.03 | 0.01–0.06 | +0.02 | 52% |
| other ship | 5,800 | 3,591 | +0.03 | 0.00–0.05 | −0.01 | 50% |

不靠模型的叠加检验：按框中心对齐，取框内条带的平均剖面，用镜像相关和质心估计图像内容相对框的位置（只取长边 ≤64 且不贴边的目标）：

| 组 | 对齐到 | 镜像 x | 质心 x | 判读 |
| --- | --- | ---: | ---: | --- |
| OGSOD bridge val | 官方框 | +1.00 | +0.89 | 内容在框右侧 |
| OGSOD bridge train | 官方框 | +0.63 | +0.50 | 方向相同 |
| OGSOD bridge val | 预测框 | −0.13 | −0.09 | 预测基本对准内容 |
| OGSOD tank val | 官方框 | −1.00 | −0.38 | 与偏移方向相反 |
| OGSOD tank train | 官方框 | +1.00 | +0.30 | 与 val 矛盾 |
| OGSOD harbor val | 官方框 | +1.88 | −1.24 | 噪声太大 |
| MSAR ship val | 官方框 | 0.00 | −0.11 | 对照，接近 0 |
| other ship val | 官方框 | +0.38 | +0.38 | 模型无偏移，内容估计仍不为 0 |

其他核对：OGSOD 和 MSAR 都是 256×256 切片；SARDet 与 OGSOD 原始标注在同一切片上 99.8% 相同（14,908 个框，四边均差约 0.001 像素）。

判读：

1. **偏移真实存在，不是巧合：** 区间窄，三类 OGSOD 一致，同尺寸的 MSAR 没有。
2. **不来自 SARDet 的转换：** 原始 OGSOD 标注已是这样。
3. **不像检测流程错误：** 只出现在 x 向和 OGSOD，翻转推理后符号不变。
4. **bridge 有不靠模型的支持；tank 没有；harbor 无法判断。**
5. **未证实：** 哪一方是“真实位置”，以及翻转是不是模型学不到偏移的原因。后者要靠 G1（关闭翻转）检验，看 dx 是否消失。

局限：叠加法受目标不对称、阴影和邻近目标影响，other ship 说明它可能有约 0.4 像素的偏差。tank 成排且集中在少数图上，叠加结果会被邻近目标污染。叠加表的区间按目标重采样，对这类聚集样本偏窄，所以这里不列。

**更正（2026-11-07）：** OGSOD 的镜像副本之间，标注恒差 2 像素，且只在翻转的轴上（[坦克诊断表 15](sardet100k_ogsod_tank_feature_diagnosis.md)）。这不依赖真实目标位置，说明至少有一部分差异来自标注或数据处理。它也解释了偏移的形状：水平翻转让相同像素对应两个相距 2 像素的答案，模型定在中间，残差约 +1（实测 +0.81）；训练没有上下翻转，y 向没有这个冲突（实测约 0）。这是推断，尚未用 G1 验证。

文献只查了一轮：最接近的是跨数据集标注对齐 LAT（不建模潜在真框）和小目标 IoU 对标注差异敏感的分析（Murrugarra-Llerena 等），未发现 SAR 上的同类工作，不能证明没有。OGSOD-1.0 的标注流程仍未读到原文，服务器上也只有它的 SAR 图。

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

E001 已于 2026-11-06 在新服务器启动（revision `8e83207`，seed 0，GPU 0、1），运行目录 `/root/zxy/experiments/deim_m_sardet100k_deim_hgnetv2_m_sardet100k_m1_geometry_36e_b32_seed0_20261106_145049_1Gh5VI`。结果待训练完成后填写。
