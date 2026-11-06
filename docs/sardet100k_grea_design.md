# GREA：保护环证据注意力

日期：2026-11-06。只针对 SARDet-100K。已实现，尚未训练。代码 `engine/deim/guard_ring_evidence.py`，配置 `configs/deim_dfine/deim_hgnetv2_m_sardet100k_grea_36e_b32.yml`，测试 `tests/test_guard_ring_evidence.py`。M1 不再作为研究方法：它修正的是标注，不是检测能力。原 M2（可学习 CFAR）并入本设计。

核心问题：**小目标和没见过的目标缺少不依赖上下文的目标证据。** 背景换成常数后，长边 <16 的已检出目标只剩 16.3%，长边 ≥16 的有 77.9%；没见过的 49.4%，见过的 68.2%。

## 诊断

E000，val，都不训练。脚本 `tools/sardet100k/detector_behaviour.py`、`context_masks.py`、`target_evidence.py`，结果在运行目录 `independent_verify_20261106/behaviour/`。

### 目标和上下文

原图 score≥0.3 检出的目标（每类抽约 150 张图），掩码后仍被检出的比例。第一次用打乱像素的掩码，伪影较强，下表只用无噪声掩码。

| 组 | 背景换常数 | 背景重度模糊 | 目标抹平（背景不变） |
| --- | ---: | ---: | ---: |
| 全部 | 61.4% | 61.4% | 7.0% |
| 见过 | 68.2% | 70.5% | 5.4% |
| 没见过 | 49.4% | 45.5% | 9.7% |
| 长边 <16 | 16.3% | 23.2% | 16.1% |
| 长边 ≥16 | 77.9% | 75.4% | 3.7% |
| aircraft | 94.7% | 90.7% | 0% |
| ship | 85.0% | 76.1% | 3.6% |
| harbor | 73.6% | 73.3% | 1.4% |
| car | 70.9% | 63.5% | 3.7% |
| car 没见过 | 60.7% | 50.3% | 5.5% |
| tank | 50.1% | 56.2% | 3.8% |
| tank 没见过 | 18.2% | 24.3% | 8.7% |
| bridge | 13.8% | 21.0% | **56.1%** |

判读：

1. 目标抹平后大多数类别几乎检不到，所以检测器确实在用目标，但小目标和没见过的目标还需要上下文才够分。
2. bridge 不同：目标抹平后仍有 56% 在原位被检出，主要靠上下文。这样的类别不应该被强迫只看目标，所以融合必须按类别自适应。
3. 掩码改变了图像统计量，也可能影响 BN 和编码器，这些数字只说明相对差异。

### 杂波亮度

- tank 在最亮的杂波三分位召回 0.69，最暗的 0.99；SCR 相同时仍是如此（0.67–0.70 对 0.95–1.00）。这和来源、目标尺寸混在一起，不能单独归因于杂波。
- ship 在高 SCR 分组里，召回随杂波变亮从 0.99 降到 0.86（最亮格只有 155 个目标）。
- 背景误检在最亮的五分位是面积占比的 1.66 倍，最暗的是 0.10 倍。最暗处多为海面，这个比较有场景差异。
- 逻辑回归中，控制 SCR、杂波、尺寸后，“是否见过”的标准化系数仍有 +0.99。

检测器不完全恒虚警，但证据比目标与上下文的结果弱。

### 否定的路线

冻结检测器，用只看目标（框加 25% 外扩）的小头输出证据，按类别与原分数做逻辑融合，系数来自另一折：

| 方案 | AP | AP50 | AP75 |
| --- | ---: | ---: | ---: |
| 基线 | 65.29 | 93.14 | 70.27 |
| 不看图的融合（对照） | 65.03 | 92.69 | 70.09 |
| 目标证据融合 | 64.34 | 93.05 | 69.50 |

事后融合不成立。局限：头只有约 3 万个训练样本，也不能改变检测器的特征。这说明简单的后处理或打分头不是答案，模块必须在训练中影响特征和匹配。

## 模块

SAR 的恒虚警检测把判断建立在三个区域上：检测单元、保护环、参考环。GREA 把这个结构放进 DETR 解码器，让每个 query 分开收集目标证据和杂波证据。

```text
backbone S2/S3（低感受野，未经过 AIFI 和 FPN）
        │
        ├─ 核心注意力：采样点限制在框内 → 目标证据 T
        ├─ 参考注意力：采样点限制在保护环以外的环带 → 杂波证据 R
        └─ 对比：C = MLP(T − R, T, R)
                 │
  query + 类别分数 → 类别自适应门控 g(T, R, C)
                 │
  原交叉注意力输出 + α · 融合证据（α 零初始化）→ 下一层
```

1. **低感受野证据源。** 诊断说明小目标依赖上下文。encoder 经过 AIFI 和 FPN 后的特征已混入大范围上下文，所以证据从 backbone 的 S2、S3 投影取，保留局部性。
2. **保护环几何。** 核心点约束在框内，参考点约束在框外 1.25–2 倍的环带，中间留保护带，防止目标能量漏进杂波估计。位置仍可学习，但范围受约束。
3. **特征域对比。** `T − R` 对应 CFAR 的检测统计量。特征域不严格满足恒虚警，所以论文只能说是受其启发。
4. **类别自适应门控。** 由 query 和类别分数决定三路权重，让 aircraft 偏向目标、bridge 偏向上下文。这是诊断 2 的直接要求，也是和普通注意力的主要区别。
5. **目标证据深监督。** 目标证据 T 单独接一个分类头，用同一匹配结果和 MAL 监督，让它必须独立分辨目标。这一路训练时使用，推理时可去掉。

只作用于最后两层解码器。原结构保留，COCO 权重可直接加载，α 零初始化，起点等于基线。

## 为什么不同于已有结果

| 已有工作 | 差异 |
| --- | --- |
| CLCR（SAR-AIRcraft test +0.09）、CLCR-Lite | 在 encoder 特征上采样，只校正分数，不更新 query，也不分开监督目标证据 |
| OST-DA（+0.64、−0.20） | 约束采样几何，但没有分开目标与杂波，也没有低感受野来源 |
| HWD、S2 注入 | 把细节加进 FPN，又混入上下文；GREA 只在 query 层面使用 |
| 目标证据融合（上表） | 事后、冻结特征；GREA 端到端 |
| CFARNets、CGMSN | 前者替换卷积层（地雷检测），后者用 CFAR 选择 YOLO 融合方式；都不是解码器 query 证据 |
| SlotSAR、IRASNet | 目标与杂波解耦，但是分类任务 |

SAR-AIRcraft 上的模块大多无效，这里也有解释：aircraft 只看目标就有 94.7%，目标证据本来就够。所以 aircraft 不是预期收益来源。

## 判定

| 项目 | 内容 |
| --- | --- |
| 协议 | 与 E000 相同，36 轮，seed 0；只看 val |
| 主指标 | val AP 比 65.29 涨 0.5 以上；不到 1.0 时要补基线和方法的第二种子 |
| 机制指标 1 | 背景换常数后，没见过的目标仍检出率高于 49.4%，长边 <16 的高于 16.3%（同一样本） |
| 机制指标 2 | 门控权重有类别差异，bridge 的目标权重低于 aircraft |
| 机制指标 3 | tank、ship 的 AP 上涨，且不靠 aircraft |
| 证伪 | AP 涨幅不到 0.5；或 AP 涨了但机制指标 1 不变，说明增益不是这个机制带来的 |
| 消融 | 基线；只加低感受野来源；加保护环；加门控；完整方法 |

## 实现要点

- **证据来源：** HGNetv2 返回 S2、S3、S4、S5，`DEIM` 把 S2、S3 给 GREA，S3、S4、S5 给编码器。S2、S3 投影到 128 维，梯度回传到骨干。
- **位置：** M 的解码器有 4 层，GREA 作用于最后两层，每层一个模块，在原交叉注意力和 FFN 之后、回归头之前，残差加到 query 上。
- **采样：** 每路 8 个点。核心点 `c + tanh(o) · 半框`，一定在框内。参考点沿 8 个固定方向，距离为 `半框 · r + 保护带`，r 在 1.25–2 之间可学习，保护带至少是 S2 的一个格子。超出图像的点不参与（全部超出时用边界值）。框停止梯度。
- **门控：** `MLP(query, 类别概率)` 输出三路 softmax，类别分数停止梯度，初始均匀。
- **深监督：** 最后一层的 T 接共享分类头，用最后一层的匹配和 focal 损失，权重 1。
- **日志：** `train_grea_alpha_*`、`train_grea_positive_c*`、`train_grea_gate_sum_c*_{target,clutter,contrast}`。门控均值 = sum 除以 positive，用于机制指标 2。
- **机制指标 1：** 训练后用 `tools/sardet100k/context_masks.py` 运行，抽样种子和 E000 相同。

文献复查（2026-11-06）：Focal DETR、Better Sampling（小目标）、Decoupled DETR 都改动解码器采样，但未把目标、保护环、参考环分开并做类别门控。两轮检索未发现相同结构，不能证明没有。

## 服务器命令（新服务器）

服务器访问 GitHub 超时时，用 Git bundle 传提交，再执行 `git merge --ff-only FETCH_HEAD`。

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
```

启动（两张卡都空闲才会启动，不排队）：

```bash
EXPERIMENT_ROOT=/root/zxy/experiments INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth RUN_PREFIX=deim_m_sardet100k bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sardet100k_grea_36e_b32.yml
```

重连：

```bash
export OUT=$(ls -dt /root/zxy/experiments/deim_m_sardet100k_*grea* | head -n 1); echo "$OUT"
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

门控和 α（最新一轮）：

```bash
python -c 'import json,os; r=[json.loads(s) for s in open(os.environ["OUT"]+"/log.txt") if s.strip()][-1]; print({k:round(v,4) for k,v in r.items() if "grea_alpha" in k}); names=["ship","aircraft","car","tank","bridge","harbor"]; [print(n,[round(r[f"train_grea_gate_sum_c{i}_{k}"]/max(r[f"train_grea_positive_c{i}"],1e-6),3) for k in ("target","clutter","contrast")]) for i,n in enumerate(names)]'
```

TensorBoard：

```bash
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

恢复（确认进程已停止）：

```bash
cd /root/zxy/deim && bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

## 局限和风险

- **没有零训练证据支持涨点。** 诊断只证明问题存在，端到端学习能否解决，要训练才知道。
- **机制指标可能只改善鲁棒性。** 背景换常数是人造条件，它变好不代表干净 AP 上涨。
- **诊断混有来源和尺寸因素。** 这里只分了类别、尺寸、见过与否。
- **新意只查过一轮文献。** 实现前要再查 DETR 里的目标与杂波分离。
- **代价：** 实现约 2–3 天（估计），一次训练约 20 小时（估计），每步会略慢。
