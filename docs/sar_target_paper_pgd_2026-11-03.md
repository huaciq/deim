# SAR-AIRcraft：以 PGD（TCSVT 2025）为对标论文的改进方向调研

日期：2026-11-03。目的：当前 v2 方案（1024 baseline 在跑）保留不动，另选一篇 2025 年 CCF-B / SCI 2 区以上的论文作为对标，
沿用它的协议，从它的问题出发做改进。相关：[协议与训练设置调研](sar_aircraft_protocol_survey.md)、[协议 v2](sar_protocol_v2.md)、
[结果台账](sar_results_ledger.md)。

## 1. 候选与选择

| 论文 | 刊物 / 等级 | 数据与协议 | 代码 | 结论 |
|---|---|---|---|---|
| **PGD**：Physics-Guided Detector for SAR Airplanes（Huang 等，arXiv 2411.12301） | IEEE TCSVT 35(12):12082–12095, 2025；CCF-B，JCR Q1 | SAR-AIRcraft-1.0，train 3047 / val 442 / test 879（与本地完全相同），1024 输入，300 轮 | [XAI4SAR/PGD](https://github.com/XAI4SAR/PGD)，YOLOv5 分支，GPL-3.0；百度网盘权重；PGSSL 预训练代码未公开（issue #10） | **选它** |
| SEFFNet：Scattering Enhancement and Feature Fusion Network（Huang 等） | IEEE TCSVT 2025（DOI 10.1109/TCSVT.2024.3470790） | SAR-AIRcraft-1.0（检索摘要） | 未找到 | 全文与数值本次未能读到（IEEE/ACM 拒绝访问），作为相关工作引用 |
| SFEF-Net | Sensors 2025（MDPI） | SAR-AIRcraft-1.0，AP50 89.4（检索摘要） | 未找到 | 刊物等级不够 |
| DiffDet4SAR | GRSL 2024 | mAP50 88.4 | 有 | 不是 2025 |
| DenoDet V2 | arXiv 2025 | 512 切片、12 轮，AP'07 69.9 | — | 未正式发表；协议不同 |

选 PGD 的理由：TCSVT 满足 CCF-B 且 JCR Q1；它是目前该数据集公认的最高结果（PGD-YOLOv8 90.7 mAP50）；划分与本地
完全一致；输入 1024 与正在跑的 v2 阶段 1 相同；代码可复现基线。

## 2. PGD 做了什么

三个模块插在 YOLO 类"骨干-颈部-头部"检测器上：

- **PGSSL**：先用 Harris-Laplace 角点 + EM 拟合 GMM，从飞机切片生成 K 通道"散射结构分布"热图作为伪标签；
  用 HRNet（PGD）或 TransPose（PGD-Lite）在外部数据（SADD、TerraSAR-X、MSAR、SAR-ACD 共 10240 个 256×192 切片）上回归
  这张热图（MSE），然后**冻结**。
- **PGFE**：以冻结 PGSSL 特征为 query、FPN 特征为 key/value 做交叉注意力，增强颈部特征（+2M 参数）。
- **PGIP**：头部额外预测单通道散射点掩码，目标是框内响应 ≥ η×局部最大值的强散射点，focal loss 监督。

主要结果（test，mAP50，单次）：YOLOv8-R50 88.7 → 90.7；YOLOv5-R50 87.9 → 90.4；YOLOv10-R50 86.7 → 87.7；
DINO-R50 85.2（作者自跑）。逐类最低的是 Boeing737（81.9）与 other（86.3），A220 / ARJ21 在 88–90。

## 3. PGD 的问题与可改进空间

按"能否写成论文主线"的价值排序。

| # | 问题 | 证据 | 改进空间 |
|---|---|---|---|
| 1 | **只验证了 YOLO 类密集检测器**，没有 DETR 类结果；PGFE 依赖 FPN、PGIP 依赖密集像素头，难以直接迁移到 query 式检测器 | 表 I 只有 YOLOv5/8/10；DINO 只作对比基线且最低之一 | 把散射物理先验做进 **query 式实时检测器**（DEIM / D-FINE）：散射点引导 query 选择 / 初始化、散射结构作为去噪或匹配的辅助目标。这是本仓库已有基座的天然方向 |
| 2 | **两阶段 + 外部数据 + 冻结大网络**，代价高 | 表 V：基线 19M / 33 FPS → PGD 48M / 8 FPS，PGD-Lite 28M / 18 FPS；PGSSL 代码未公开 | 端到端、无外部预训练、推理零开销或近零开销（只在训练期用的辅助监督），主打"实时 + 物理引导" |
| 3 | **伪标签是手工阈值**：Harris 角点 + GMM，固定参数（≥4 点、Σ0=diag(2,2)、3σ 截断、η），无敏感性分析 | 文中只称"不敏感" | 由数据自适应产生散射结构（例如用检测器自身注意力 / 可学习关键点聚合在线更新），或对伪标签加置信度加权 |
| 4 | **只报 mAP50**，不关心定位质量 | 全文无 IoU 阈值说明，无 AP75 / AP50:95 | 我们的诊断显示型号混淆与定位都是缺口；DEIM 的 FDR 细粒度回归是强项，同时报 AP50 与 AP50:95 能形成差异化 |
| 5 | **细粒度型号混淆仍是主要误差**，模块与"型号区分"无直接关系 | Boeing737 81.9、other 86.3；PGIP 是类别无关的散射点掩码 | 类别相关的散射结构（不同机型发动机 / 尾翼散射点相对位置不同）：结构感知的分类分支或对比约束 |
| 6 | **实验严谨性**：单次运行无方差；消融用 50 轮、主表用 300 轮，消融最佳 88.0 对不上主表任何一行；部分基线抄自其它文献 | 表 IV vs 表 I；表 II 注记 | 我们在同一协议下多种子、消融与主表同一日程，本身就是可写进论文的优势 |
| 7 | 外部预训练数据与 SAR-AIRcraft（同为高分三号）是否有场景重叠未说明 | 文中未提 | 我们的方法不用外部数据，避免这一质疑 |

建议的论文主线（1+2+5 组合）：**面向实时 DETR 的物理散射引导细粒度 SAR 飞机检测**——训练期用散射结构作辅助监督与
query 引导，推理不增加或几乎不增加开销，在 PGD 协议下与 PGD / YOLOv8 / DINO 同表比较，同时报 AP50 与 AP50:95。
已有的 REM（102 轮 640 下 test +1.57 AP）可作为特征侧模块候选，在新协议下重新验证。

## 4. 统一协议（照搬 PGD）

| 项目 | PGD | 我们照做 | 备注 |
|---|---|---|---|
| 划分 | train 3047 / val 442 / test 879 | 相同（即方案 A） | 已就绪 |
| 输入 | 1024×1024 训练与测试 | 1024 | 与 v2 阶段 1 一致 |
| 轮数 | 300 | 需决定：300 轮对齐，或在更短日程下报告并注明 | 1024、全局 batch 12 时每轮约 254 iter |
| 指标 | mAP（由基线数值反推为 mAP50，101 点） | DEIM COCO 评估索引 1（AP50）为主，索引 0 为辅 | 与官方、PGD 同口径 |
| 报告集合 | test | test；val 选权重 | |

## 5. 最大风险与第一步

**我们的 baseline 和 PGD 表中的基线差距过大。** 640 下 DEIM-M test AP50 65–67，而 PGD 表中 YOLOv8-R50 为 88.7、DINO 85.2。
如果 1024 baseline 跑完 AP50 仍远低于 85，那么即使模块有 +2 的增益，也无法和 PGD 同表比较，选题就站不住。

所以第一步不是写新模块，而是**校准**：

1. 等 v2 阶段 1 的 1024 baseline 出结果，看 test AP50。
2. 在旧服务器上用 PGD 官方仓库原样跑一次它的 YOLOv5 baseline（`resnet18_RepPAN.yaml`，300 轮），用我们的 test.json
   和 DEIM 同一个 COCO 评估脚本重新评估它的预测。目的是确认 88 左右是否在同一口径下可复现，以及差距来自
   评估口径、训练日程还是检测器本身。
3. 根据 1、2 的结果决定：差距 ≤3 → 直接以 PGD 为对标推进；差距大 → 先查清原因（例如 YOLO 的 val 设置、
   NMS / 置信度阈值、是否用了 val 训练），再决定是否换对标论文。

## 6. SEFFNet（TCSVT 35(2):1936–1948, 2025）核读

用户提供全文 PDF（`C:\project\文献\2025-2026_跨分辨率相似飞机\`）。作者 Huang、Zhang 等（上海交大），基线 YOLOv5s + CSPDarknet53（加 C6 层）。

| 模块 | 做法 | 性质 |
|---|---|---|
| SIEEM | 输入级预处理：结构张量（Harris 同源）的迹 H 与行列式 K，H ≤ T（T=0.8）判为平坦并置 0，其余保留原值；三通道复制 | 无参数、无需训练、离线可算 |
| SDCAM | SPD-Conv（space-to-depth）+ 坐标注意力 | 通用模块组合 |
| FFP | PANet + BiFPN 式跳连 + ASFF 式可学习权重；后接 GAM | 通用模块组合 |
| CFH | 相邻三层上下采样相加的上下文头 | 通用模块组合 |

协议与结果：

- SAR-AIRcraft-1.0 **当作单类**，**随机 4:1 划分**（train 3495 / test 873，不是官方划分），640 输入，batch 8，SGD lr 0.01。
- SAR-AIRcraft 上**只报 Precision / Recall / F1**，没有 AP：SEFFNet P 0.910 / R 0.945 / F1 0.927；YOLOv5 F1 0.859，YOLOv8 0.882。
- 消融全部只在 SADD（224 切片）上做，SADD 上 SEFFNet APs 0.971 vs YOLOv5 0.951。SIEEM 的阈值 T 在 0.6–1.0 之间
  F1 只变 0.957–0.963。
- 单次运行，无方差；没有找到公开代码。

**结论：不适合作对标论文，但 SIEEM 可以作为 PGSSL 的廉价替代。**

- 协议不统一：单类、随机划分、无 mAP，无法与我们 7 类、官方划分、AP50 的结果放进同一张表，失去"统一协议"的好处。
- 后三个模块是 SPD-Conv / CA / BiFPN / ASFF / GAM 的组合，新意弱；本仓库已试过 space-to-depth 类改动（REM space-to-depth
  test −1.93，SDCP −2.01），预期收益低。
- SIEEM 在输入端把"平坦"像素硬置 0，会丢掉跑道、停机位等上下文，也偏离 COCO 预训练的输入分布；对 7 类型号识别不一定有利。
  但它说明**结构张量响应本身就是一张无需学习的散射先验图**。

## 7. 合并方案：PGD 协议 + 结构张量先验代替 PGSSL

PGSSL 的难点是外部数据、未公开的伪标签与预训练代码、冻结 HRNet 的推理代价。PGSSL 学到的东西本质上就是"Harris 角点 + GMM"
得到的散射结构热图，而 SIEEM 证明同类结构张量响应可以直接解析计算。所以：

- **协议**：照搬 PGD（官方划分、1024、test AP50 为主，AP50:95 为辅）。对比表引用 PGD 与 SEFFNet 两篇 TCSVT。
- **先验**：用结构张量 / Harris 响应（加框内 η×局部最大值的强散射点筛选，即 PGIP 的标签规则）离线生成散射图，
  不需要外部数据和预训练网络。
- **复现对照（"PGD*"）**：D-FINE + PGIP 式辅助头（训练期）+ 用解析散射图代替 PGSSL 特征做 PGFE 交叉注意力。
- **本文方法**：在此基础上做 DETR 专用部分——散射图引导 encoder top-k query 选择；强散射点引导 decoder 可变形注意力采样点，
  针对型号区分；推理期尽量零开销。

## 来源

- PGD arXiv：https://arxiv.org/abs/2411.12301 ；代码：https://github.com/XAI4SAR/PGD ；发表信息：https://pure.nwpu.edu.cn/en/publications/physics-guided-detector-for-sar-airplanes/
- SEFFNet 条目：https://dl.acm.org/doi/abs/10.1109/TCSVT.2024.3470790 （未能读取全文）
- SFEF-Net：https://www.mdpi.com/1424-8220/25/10/2988
- DiffDet4SAR：https://arxiv.org/abs/2404.03595
- DenoDet V2：https://arxiv.org/abs/2508.09392
