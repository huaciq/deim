# SARDet-100K 剩余缺口与上升空间：文献事实与本项目诊断（截至 2026-11-04）

阅读约定：
- "Cited Findings" 下均为有来源的事实，来源为论文表/节号或本地 `文件:行号`。
- "Inferences" 下均为本笔记的推断或自行计算，不是任何来源的原话。
- 本地文件路径相对 `C:\project\deim`。
- 所有审计类实验都只作内部诊断，不改官方划分；项目约定见 `C:\Users\Admin\.claude\projects\C--project-deim\memory\official-split-unchanged.md`（该记录针对 SAR-AIRcraft，这里按同样原则对待 SARDet-100K，是否适用由用户确认）。
- 按 `AGENTS.md` 的约定，重复检测不使用任何哈希比对，只用相关性匹配。

## 问题一：数据集结构、切片、划分共享，以及重复、泄漏和标注噪声的报告

### Takeaway
SARDet-100K 由 10 个来源合并，四个来源占训练图 92.7%，类别与来源强绑定（car 只来自 SIVED，harbor 只来自 OGSOD）。论文本身没有审计重复和跨划分泄漏，但 RSAR 明确说 SARDet-100K 仍有重复和无标注图，并为避免泄漏做了清洗。所有公开资料都没有给出重复率的具体数字。

### Cited Findings

**来源与划分（SARDet-100K Table 1、Table 2）**

| 来源 | 目标 | 分辨率 (m) | 波段 | 卫星/传感器 | 图像 train/val/test | 实例 train/val/test |
|---|---|---|---|---|---|---:|
| AIR_SARShip 1*（512 切片） | S | 1、3 | C | GF-3 | 438 / 23 / 40 | 816 / 33 / 209 |
| AIR_SARShip 2 | S | 1、3 | C | GF-3 | 270 / 15 / 15 | 1,819 / 127 / 94 |
| HRSID | S | 0.5~3 | C/X | S-1B、TerraSAR-X、TanDEM-X | 3,642 / 981 / 981 | 11,047 / 2,975 / 2,947 |
| MSAR*（512 切片） | A,T,B,S | ≤1 | C | HISEA-1 | 27,159 / 1,479 / 1,520 | 58,988 / 3,091 / 3,123 |
| SADD | A | 0.5~3 | X | TerraSAR-X | 795 / 44 / 44 | 6,891 / 448 / 496 |
| SAR-AIRcraft*（512 切片） | A | 1 | C | GF-3 | 13,976 / 1,923 / 2,989 | 27,848 / 4,631 / 5,996 |
| ShipDataset | S | 3~25 | C | S-1、GF-3 | 31,784 / 3,973 / 3,972 | 40,761 / 5,080 / 5,044 |
| SSDD | S | 1~15 | C/X | S-1、RadarSat-2、TerraSAR-X | 928 / 116 / 116 | 2,041 / 252 / 294 |
| OGSOD | B,H,T | 3 | C | GF-3 | 14,664 / 1,834 / 1,833 | 38,975 / 4,844 / 4,770 |
| SIVED | C | 0.1、0.3 | Ka,Ku,X | 机载 SAR 合成切片 | 837 / 104 / 103 | 9,561 / 1,222 / 1,230 |
| **合计** | 6 类 | 0.1~25 | | | **94,493 / 10,492 / 11,613** | **198,747 / 22,703 / 24,023** |

以上来自 [SARDet-100K Table 1、Table 2](https://arxiv.org/html/2403.06534v3)，其 PDF 文本已核对分辨率和目标列（[PDF](https://arxiv.org/pdf/2403.06534)）。

**切片与划分规则**
- 超过 1000×1000 的图做切片；AIR_SARShip 1、MSAR、SAR-AIRcraft 切成 512×512，重叠 200 像素；Table 1 脚注标记了这三个来源 — [SARDet-100K App. A.2、Table 1](https://arxiv.org/pdf/2403.06534)。
- 来源自带划分时沿用，否则按 8:1:1 划分 — [SARDet-100K App. A.2](https://arxiv.org/pdf/2403.06534)。
- 论文正文没有提到"按场景分组"或"同场景切片不跨划分" — [SARDet-100K App. A.2 全文](https://arxiv.org/html/2403.06534v3)。其余来源的切片尺寸论文没有逐一列出。
- 本地筛选验证集 4,096 张的实际尺寸：256×256 共 2,815 张，512×512 共 827 张，800×800 共 382 张，224×224 共 18 张，1000×1000 共 5 张，非正方形 49 张 — `docs/sardet_offset_check.md:371-374`。

**类别与尺寸分布**
- 实例占比：ship 47%、aircraft 21%、bridge 14%、tank 11%、car 5%、harbor 2% — [SARDet-100K Fig. S6(b)](https://arxiv.org/pdf/2403.06534)（从 PDF 文本层读取）。
- 平均实例面积（源像素）：harbor 7,869、aircraft 5,490、car 与 bridge 约 1,781–1,791（PDF 文本层两次渲染的顺序相反，无法区分哪个对应哪个）、ship 1,315、tank 461 — 同上 Fig. S6(b)；tank 的 461 另见 [App. A.8](https://arxiv.org/pdf/2403.06534)。
- "Tank"指储油罐，不是军用坦克 — [Issue #33 仓库作者回复](https://github.com/zcablii/SARDet_100K/issues/33)。
- 640 输入下面积 <32² 的小目标占比：全量训练 23.9%、完整验证 25.8% — `docs/sardet_screening_handoff.md:36-37`。标准 COCO APs 按原始标注面积计算，两种统计不能混用 — `docs/sardet_screening_handoff.md:38-39`。
- 官方 JSON 使用匿名数字文件名，没有来源字段 — `docs/sardet_screening_handoff.md:41-44`。

**重复、泄漏与噪声的公开报告**
- RSAR §4.1 写道："duplicate and unannotated images remain"。作者对重复图只保留标注最多的一张，删除无标注图，并说明目的是防止数据泄漏造成评估偏差 — [RSAR §4.1](https://arxiv.org/pdf/2501.04440)。RSAR 没有报告重复率。
- RSAR 还排除了部分方向过于模糊、无法准确标注的飞机数据；val 和 test 不参考弱监督模型，全部手工标注 — [RSAR §4.1](https://arxiv.org/pdf/2501.04440)。
- RSAR 规模：95,842 张、183,534 个实例；类别占比 ship 62.19%、bridge 18.57%、car 6.55%、tank 6.19%、aircraft 4.26%、harbor 2.24% — [RSAR §4.2、Fig. 4(c)](https://arxiv.org/pdf/2501.04440)。
- 仓库 README 称 RSAR 是 SARDet-100K 的"近似旋转标注版本" — [README](https://github.com/zcablii/SARDet_100K)。
- Issue #3：标注中存在面积为 0 的框，例如 `bbox [191,1,47,0]`。作者解释这来自切片撕裂，训练配置默认过滤小于 16 像素的实例，删除与否可选 — [Issue #3](https://github.com/zcablii/SARDet_100K/issues/3)。
- Issue #13：用户指出单目标来源的图里可能有未标注的其他类别（如飞机图中的车）。作者的回应是这类目标在该分辨率下通常太小，并引用论文"类别无冲突"的说法；帖子没有说明做过逐图检查 — [Issue #13](https://github.com/zcablii/SARDet_100K/issues/13)。
- Issue #44：用户质疑极噪图（0041963.jpg）。作者确认其很噪，但称为保持多样性有意保留，且 ship 标注准确 — [Issue #44](https://github.com/zcablii/SARDet_100K/issues/44)。
- Issue #35：作者指向一个 `mapping.json` 作为 SSDD 和 HRSID 划分的依据，帖子没有说明其内容 — [Issue #35](https://github.com/zcablii/SARDet_100K/issues/35)。
- 全部 62 个 issue 的标题中没有以 duplicate 或 leakage 为主题的 — [Issues 第 1 页](https://github.com/zcablii/SARDet_100K/issues?q=is%3Aissue)、[第 2 页](https://github.com/zcablii/SARDet_100K/issues?page=2&q=is%3Aissue)、[第 3 页](https://github.com/zcablii/SARDet_100K/issues?page=3&q=is%3Aissue)。我只读了 #3、#13、#18、#31、#33、#35、#44 的正文。
- test 集：README 仍写只发布 train 和 val（[README](https://github.com/zcablii/SARDet_100K)），但 Issue #31 在 2024-08-31 公布了 test 下载链接（[Issue #31](https://github.com/zcablii/SARDet_100K/issues/31)）。两处矛盾，以后者较新。
- 2025 年的 SAR ATR 综述只把 SARDet-100K 列入数据集表和结果表，没有谈其质量问题；一般性地提到弱散射、叠掩、遮挡造成漏标和错标，并批评高准确率常反映针对数据集的优化，统一评测基准缺失 — [arXiv 2509.22159 Table III、Table V、Sec. II-B、V、VI-A](https://arxiv.org/html/2509.22159)。

**本项目关于来源的已知事实**
- SAR-AIRcraft 原始数据为 4,368 张、16,463 个目标 — `docs/sar_aircraft_dataset.md:166`。
- CPU 检查显示 SAR-AIRcraft 官方 val 有 82%、test 有 50% 的目标与 train 目标近似相同 — `C:\Users\Admin\.claude\projects\C--project-deim\memory\official-split-unchanged.md`；另见 `docs/sar_protocol_b.md:58`。
- OGSOD 原验证集 3,667 张 — `docs/sardet_offset_check.md:378`。
- SARDet-100K 的来源包含 OGSOD，不能当作独立外部数据 — `docs/sardet_screening_handoff.md:42-43`。

### Inferences
- **来源集中（自行计算）**：ShipDataset、MSAR、OGSOD、SAR-AIRcraft 四个来源占训练图 92.7%。总体 mAP 主要反映这四个来源。
- **划分沿用（推断）**：OGSOD 的 1,834 + 1,833 = 3,667，恰好等于原验证集张数，说明原 val 被对半分成 SARDet 的 val 和 test。HRSID 的 981 + 981 同样像把原 test 对半分。这些数字支持"沿用后再切分"，但论文没有明说。
- **切片放大（自行计算）**：SAR-AIRcraft 从 4,368 张、16,463 个实例变成 18,888 张、38,475 个实例，实例数放大 2.34 倍。重叠 200 像素使同一实例在多个切片中重复出现。
- **验证集泄漏下限（估算）**：SAR-AIRcraft 占 val 实例 20.4%（4,631/22,703）。如果 82% 的近似重复率能迁移到切片层，仅这一个来源就有约 16.7% 的 val 实例与 train 近似重复；test 约 12.5%（25.0% × 50%）。这是粗略估计：82% 是在原始切片上量的，没有在 SARDet 切片上量过。
- **随机划分来源（推断）**：ShipDataset（31,784/3,973/3,972，正好 8:1:1）若按切片随机划分，同场景切片必然跨划分。论文未说明，属于需要验证的假设。
- **RSAR 删除了什么（自行估算）**：相对 SARDet-100K，图像少 20,756 张（−17.8%），实例少 62,119 个（−25.3%）。按占比折算：aircraft 约 51.6k→7.8k，tank 约 27.0k→11.4k，ship、bridge、car、harbor 基本不变。减少量约 96% 落在 aircraft 和 tank。SARDet 的占比是取整值，每类误差约 ±1.2k。RSAR 没有把这些删除分别归因于重复、无标注还是方向模糊，所以不能把 25% 当作重复率。
- **类别与来源混淆（推断）**：car 只来自 SIVED，harbor 只来自 OGSOD。harbor 占比最小（2%），AP50 却最高（见问题二），这可能部分来自模型学到了来源风格，而不是类别本身。
- **APs 的含义混杂（推断）**：COCO 面积按源像素计算，而不同来源缩放倍数不同（640 输入时 256 图放大 2.5 倍，800 图缩小到 0.8 倍）。SARDet 的 APs 不是"输入上看到的小目标"的 AP。

### Gaps
- 没有找到任何公开的 SARDet-100K 重复率、跨划分近似重复率或按场景分组的统计。
- `mapping.json` 托管在校内 SharePoint，未读取，不知道它能否给出逐图来源。
- 除三个切片来源外，其余来源的切片与重叠设置，论文没有列出；本笔记未去逐个核对各来源的原始论文。

## 问题二：已发表的逐类、逐尺寸 AP，AP50→AP75 落差，以及有无错误分解

### Takeaway
所有全文表格里 bridge 和 tank 都是最低或次低的类别，bridge 的 AP75 崩塌最严重。AP50−AP 在各方法间稳定在 29–37 点，是剩余空间的最大部分，所以定位是主要损失的说法有数据支持。但没有找到任何 SARDet-100K 上的 TIDE 式错误分解，"定位"与"标注噪声"目前也无法区分。

### Cited Findings

**总体与逐尺寸（差值列自行计算）**

| 方法 | 来源 | 协议 | AP | AP50 | AP75 | APs | APm | APl | AP50−AP75 | AP50−AP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN + ConvNeXt-B + MSFA | [SARDet-100K Table S9](https://arxiv.org/html/2403.06534v3) | 800 输入，12e，test（[App. A.6](https://arxiv.org/pdf/2403.06534)） | 56.4 | 88.2 | 61.5 | 51.1 | 68.3 | 62.4 | 26.7 | 31.8 |
| GFL | [DenoDet V2 Table S7](https://arxiv.org/pdf/2508.09392) | 512 输入，12e，划分未明说 | 55.01 | 85.16 | 58.87 | 49.44 | 67.29 | 60.45 | 26.29 | 30.15 |
| Deformable DETR | 同上 | 同上 | 52.00 | 88.77 | 54.03 | 46.99 | 63.58 | 58.55 | 34.74 | 36.77 |
| AutoAssign | 同上 | 同上 | 53.95 | 89.58 | 55.96 | 50.14 | 63.40 | 54.73 | 33.62 | 35.63 |
| DenoDet V2 | 同上 | 同上 | 56.71 | 85.86 | 61.55 | 51.45 | 68.75 | 61.34 | 24.31 | 29.15 |
| ViT-B + MAE 预训练 | [arXiv 2501.11249 Table I](https://arxiv.org/html/2501.11249v1) | val | 54.9 | 85.9 | 60.5 | 50.0 | 65.3 | 58.7 | 25.4 | 31.0 |
| RT-DETR（R18） | [MDIS-DETR Table 8](https://doi.org/10.3390/rs18132076) | 640 输入，150e，无预训练，无显式增强，划分未明说 | 54.24 | 85.89 | 58.41 | 48.74 | 67.72 | 62.54 | 27.48 | 31.65 |
| MDIS-DETR（150 FPS） | 同上 | 同上 | 58.82 | 89.31 | 63.68 | 53.43 | 71.37 | 67.08 | 25.63 | 30.49 |
| 本地 DEIM-D-FINE-S，10% 子集，旧协议（日志） | `docs/sardet_offset_check.md:351-353` | 640 输入，120e，筛选 val | 42.75 | 72.63 | 44.59 | 35.78 | 50.38 | 43.10 | 28.04 | 29.87 |
| 本地 DEIM-D-FINE-S，10% 子集，新协议（日志） | `docs/fdr_variance_step0.md:584-586` | 同上 | 43.06 | 72.83 | 45.10 | 35.74 | 51.02 | 44.95 | 27.73 | 29.77 |

**逐类 AP（@[.50:.95]）**

| 方法 | ship | aircraft | car | tank | bridge | harbor | 来源 |
|---|---:|---:|---:|---:|---:|---:|---|
| Faster + ConvNeXt-B + MSFA | 66.9 | 45.5 | 65.5 | 45.4 | 44.1 | 71.1 | [Table S13](https://arxiv.org/pdf/2403.06534) |
| DenoDet V2 | 64.85 | 58.08 | 63.26 | 46.18 | 37.66 | 70.22 | [Table S7](https://arxiv.org/pdf/2508.09392) |
| RT-DETR | 65.94 | 48.16 | 62.61 | 43.33 | 38.79 | 66.63 | [MDIS-DETR Table 8](https://doi.org/10.3390/rs18132076) |
| MDIS-DETR | 68.69 | 53.63 | 67.02 | 50.48 | 42.62 | 70.47 | 同上 |

Table S13 的 aircraft（45.5）与 DenoDet V2 的 aircraft（58.08）差 12.6 点，来源分别是 test 800 输入和划分未明说的 512 输入。

**唯一公开的逐类 AP50/AP75/逐尺寸表（[SARDet-100K Table S13](https://arxiv.org/pdf/2403.06534)，ConvNeXt-B + MSFA）**

| 类别 | AP | AP50 | AP75 | APs | APm | APl | AP50−AP75（自算） |
|---|---:|---:|---:|---:|---:|---:|---:|
| ship | .669 | .923 | .783 | .653 | .714 | .555 | .140 |
| aircraft | .455 | .753 | .466 | .419 | .458 | .47 | .287 |
| car | .655 | .985 | .792 | .553 | .674 | n/a | .193 |
| tank | .454 | .766 | .427 | .429 | .891 | n/a | .339 |
| bridge | .441 | .885 | .368 | .411 | .609 | .714 | **.517** |
| harbor | .711 | .979 | .858 | .601 | .751 | .756 | .121 |

- 作者自述性能较低的类型：小目标（tank）、大长宽比（bridge）、外观变化大（airplane）。他们也提到模型在小而密的目标上漏检，图像质量差时漏检更多，困难样本定位不准，但未量化 — [SARDet-100K App. A.8](https://arxiv.org/pdf/2403.06534)。
- RSAR 的旋转框 AP50（Table 11）中 tank 明显最低，FCOS 也只有 33.9，其他方法 21.7–30.8 — [RSAR Table 11](https://arxiv.org/pdf/2501.04440)。RSAR 以 AP75 为主指标，因为它更能反映角度精度 — [RSAR §5](https://arxiv.org/pdf/2501.04440)。

**错误分解**
- 已读全文的论文（[SARDet-100K](https://arxiv.org/html/2403.06534v3)、[DenoDet V2](https://arxiv.org/pdf/2508.09392)、[MDIS-DETR](https://doi.org/10.3390/rs18132076)、[MSPaDet](https://doi.org/10.3390/rs18101513)）都没有 TIDE 式分解；SARDet 论文 App. A.8 只有定性的失败案例描述。另做了一轮专门检索，也只返回数据集论文本身。这只说明没有检索到，不能证明不存在。

**不可比与数据可信度问题**
- 12 轮的基线表（MSFA、DenoDet）与长训练结果差距很大：SARLite 用 YOLO11s 系，SGD、200 轮、批量 64，报告 YOLO11-s 64.0、YOLOv10-s 65.1、本文 67.7 — [SARLite 正文与表](https://pmc.ncbi.nlm.nih.gov/articles/PMC13276379/)。SAR-NAS 报告 YOLOv10-S 68.84、SAR-NAS-S 69.27 — [SAR-NAS Table 1、2](https://arxiv.org/pdf/2509.01279v1)。
- SARLite 的消融实验用 5 折划分，把数据重新分成互不重叠的折，而不是官方划分 — [SARLite Implementation](https://pmc.ncbi.nlm.nih.gov/articles/PMC13276379/)。
- 同一 Faster R-CNN R50，12 轮 ImageNet 初始化 mAP 49.0，36 轮直接微调 52.8 — [SARDet-100K Table S8、S11](https://arxiv.org/html/2403.06534v3)。
- MSPaDet Table 4 的多行基线与 DenoDet V2 Table S7 逐项恰好差 0.30，例如 GFL 54.71/84.86/58.57/49.14/66.99 对 55.01/85.16/58.87/49.44/67.29，YOLOX 33.78/66.47/31.01/28.19/42.76 对 34.08/66.77/31.31/28.49/43.06 — [MSPaDet Table 4](https://doi.org/10.3390/rs18101513)、[DenoDet V2 Table S7](https://arxiv.org/pdf/2508.09392)。
- MDIS-DETR Table 8 中大部分基线数值与 DenoDet V2 Table S7 相同；DenoDet V2 一行为 56.67/86.24/61.02（原文 56.71/85.86/61.55），Conditional DETR 的 AP75 为 51.67（原文 44.40）— [MDIS-DETR Table 8](https://doi.org/10.3390/rs18132076)、[DenoDet V2 Table S7](https://arxiv.org/pdf/2508.09392)。

### Inferences
- **最低类别（自行计算）**：DenoDet V2 Table S7 全部 28 个方法的类别均值：bridge 32.6、tank 37.4、aircraft 54.6、harbor 58.1、ship 59.2、car 59.8。其中 mAP≥52 的 17 个方法：bridge 36.4、tank 42.8、aircraft 56.2、ship 62.1、car 62.1、harbor 63.1。排序稳定，bridge 和 tank 是真正的低点。
- **逐尺寸（自行计算）**：同一批 17 个方法的均值 APs 48.3、APm 65.4、APl 57.9。上表全部方法和本地子集都满足 APs < APl < APm。APl 低于 APm 是稳定现象，值得单独解释（切片截断大目标是待验证假设）。
- **定位是最大单项，但有上限未知（推断）**：MDIS-DETR 距 100 分差 41.18，其中 AP50−AP 占 30.49（74%）；DenoDet V2 占 67%；RT-DETR 占 69%。AP50−AP 在不同框架下只有 29–37 点的范围，且 12 轮和 150 轮的方法相差不大。
- **AP75 落差不均匀（推断）**：bridge 的 AP50 .885 很高，AP75 只有 .368，说明它能被找到但框不准；ship 和 harbor 落差很小。aircraft 和 tank 的 AP50 也低（.753、.766），问题不只是定位，还有检出和误检。
- **稀有类得分最高（推断）**：harbor 只占 2%、car 只占 5%，AP50 却分别是 .979、.985。长尾不是这个数据集的主要矛盾，反而可能是"类别即来源"的混淆在抬高分数。
- **表格可靠性（推断）**：差 0.30 的现象和大量相同的基线行说明，这些表不能算独立复现。比较方法时应当只信同一论文内自己实际跑过的结果，且最好自己复跑。我不对原因下判断。
- **排行榜不可比（推断）**：YOLO 系 64–69 对 MMDetection 12 轮系 52–58、RT-DETR 150 轮 54 的差距，过大，无法用模型解释。可能原因包括划分（5 折或其他）、评测实现（Ultralytics 内置 mAP 与 COCOeval 的差异）、训练长度、小框过滤。未核实。论文写 "SOTA" 时必须先固定协议。
- **训练长度的作用（推断）**：R50 上 12→36 轮带来 +3.8，比 DenoDet V2 相对 GFL 的 +1.70 还大。方法增益容易被训练长度盖住。
- **aircraft 差 12.6 点（推测，证据弱）**：test 上 SAR-AIRcraft 的"见过"比例（50%）低于 val（82%），可能使 aircraft 在 test 上更低；也可能只是输入尺寸或方法差异。按来源评测才能判断。
- **表内错误**：Table S13 的 Average 行 @m 为 .733，而六类平均是 .683，也与 Table S9 的 68.3 一致，.733 疑为笔误（自行核算；其余列的平均都能对上）。

### Gaps
- 公开文献只有 Table S13 一张逐类 AP50/AP75/逐尺寸表，来自一个 Faster R-CNN 模型。其他方法没有逐类 AP75。
- DenoDet V2、MDIS-DETR、MSPaDet 没有在我读到的文本中明确写出 SARDet-100K 指标用的是 val 还是 test。
- SARLite、SAR-NAS 与其他表的巨大差距没有解释；这两篇的评测实现未细查。
- PDF 表格是从文本层抽取的，MSPaDet Table 4 版式混乱，只比对了可对齐的行。

## 问题三：这些方向已有人做了吗

### Takeaway
小目标、斑点/频域去噪和实时性已经有不少工作，竞争激烈。定位精度未被当作专门目标研究，跨来源泛化没有严格协议，标注噪声和错误分解在 SARDet-100K 上都没有检索到工作。

### Cited Findings
- **定位精度 / IoU 级指标**：DenoDet V2、MDIS-DETR 报告 AP75，但都不把它作为设计目标 — [DenoDet V2](https://arxiv.org/pdf/2508.09392)、[MDIS-DETR §4](https://doi.org/10.3390/rs18132076)。RSAR 把 AP75 作主指标，服务于旋转角度 — [RSAR §5](https://arxiv.org/pdf/2501.04440)。本次检索未见针对 SARDet-100K 高 IoU 定位的工作。
- **小目标**：MSPaDet 声称相对最强对手 APs +0.3，并可视化小于 5×5 像素的目标 — [MSPaDet §4.2.3、§4.4.1](https://doi.org/10.3390/rs18101513)。MDIS-DETR APs 53.43 — [MDIS-DETR Table 8](https://doi.org/10.3390/rs18132076)。
- **跨来源 / 跨传感器**：MSPaDet 在 SAR-AIRcraft 训练、在 SARDet-100K 的飞机子集上直接测，AP 0.4101、召回 0.61，并明确说这不是严格的留一传感器评估，因为目标数据集没有可靠的逐图传感器标签 — [MSPaDet §4.2.4、Table 5](https://doi.org/10.3390/rs18101513)。SARLite 同时跑 SSDD 称为跨数据集泛化，但只是各自训练 — [SARLite](https://pmc.ncbi.nlm.nih.gov/articles/PMC13276379/)。SED-FOD 做少样本跨传感器检测，但用 FARAD 和 MiniSAR，不是 SARDet-100K — [SED-FOD 摘要与 §4](https://arxiv.org/pdf/2608.18755)。一篇 2026 年舰船分类研究显示 SARDet-100K 预训练在域内最好，跨数据集掉得很多，但这是分类 — [arXiv 2609.31658 检索摘要](https://arxiv.org/abs/2609.31658)（只读了摘要）。
- **斑点鲁棒**：DenoDet V2 相位-幅度交叉去噪 — [DenoDet V2](https://arxiv.org/pdf/2508.09392)。MSPaDet 用 DTCWT 相位一致性，并把"与去斑流水线对比"单独列节 — [MSPaDet §3、§4](https://doi.org/10.3390/rs18101513)。MDIS-DETR 多域干扰抑制 — [MDIS-DETR](https://doi.org/10.3390/rs18132076)。SARMAE 是噪声感知的自监督预训练 — [SARMAE 仓库](https://github.com/mililab/sarmae)（只读了检索摘要）。这些都在加模块或预训练，本次检索未见干净/加噪成对的测试协议。
- **实时**：MDIS-DETR 150.17 FPS、58.82 mAP；RT-DETR 187.93 FPS、54.24 mAP（单张 RTX 4090）— [MDIS-DETR Table 8、§4.2](https://doi.org/10.3390/rs18132076)。SAR-NAS-N 1.97M 参数 60.18 — [SAR-NAS Table 1](https://arxiv.org/pdf/2509.01279v1)。SARLite 7.8M 参数 67.7 — [SARLite](https://pmc.ncbi.nlm.nih.gov/articles/PMC13276379/)。边缘版 YOLO 1.9M 参数、54.0 mAP50-95 — [Remote Sens. 17(13):2168 检索摘要](https://www.mdpi.com/2072-4292/17/13/2168)（只读了摘要）。
- **标注噪声**：SAR 噪声标签的工作集中在分类，如 CLSDF（MSTAR）— [arXiv 2508.07656](https://arxiv.org/pdf/2508.07656)。本地调研另记录了 SFEF-Net 的 NR 损失，用在 SAR-AIRcraft 的 YOLO 上 — `docs/sar_mechanism_survey_2026-11-03.md:48`。本次检索未见 SARDet-100K 上的标注噪声工作。

### Inferences
- **拥挤度（推断）**：斑点/频域模块方向已很拥挤，边际增益约 1–2 点。DenoDet V2 相对 GFL +1.70，MDIS-DETR 相对 RT-DETR +4.58（而 RT-DETR 基线没有预训练和增强，这个增益可能被高估）。
- **MSPaDet 的跨域结果有污染（推断）**：SARDet-100K 的飞机子集本身包含 SAR-AIRcraft 切片（[Table 1](https://arxiv.org/html/2403.06534v3)），所以这个 0.41 不是纯粹的跨数据集泛化。
- **真正的空白（推断）**：严格的留一来源协议、逐来源 AP、重复感知评测、错误分解，都是低成本且没人做的基础工作。
- **DEIM/D-FINE 的位置（推断）**：本次未检索到 D-FINE 或 DEIM 在 SARDet-100K 全集上的公开结果。检索有限，不能断言不存在。

### Gaps
- 一篇 AAAI 论文（检索命中 `ojs.aaai.org/.../37761/41723`）未能下载；sota2 排行榜返回 500，未读取。
- 2026 年的 MDPI 类论文数量多，本次只读了两篇全文，不能保证穷尽。

## 问题四：本项目的证据以及与文献的对应

### Takeaway
本地已有的 SARDet 证据全部来自 10% 子集（S 模型，筛选 val 4,096 张），文档中没有全集 baseline 的任何结果。子集的 AP50 比全量文献低约 13–16 点，但 AP50−AP 与文献几乎相同。σ 确实随误差单调变化，但四边各向同性，和分数高度共线；σ 是否有额外排序价值在文档中尚未测量。

### Cited Findings

**用户引用的基线数字需要澄清**
- 42.75 是旧协议日志 AP，而 72.73 和 44.81 是导出重算值。旧协议日志为 AP 42.7546 / AP50 72.6263 / AP75 44.5886；导出重算为 AP≈42.88 / AP50 72.73 / AP75 44.81；两者差 +0.1254 AP，原因是导出器按 query 取最大类别并使用分数阈值 — `docs/sardet_offset_check.md:351`、`:383-393`。
- 这条是旧协议（warmup=1000、flat=60、no_aug=8），不是后来的固定协议 — `docs/sardet_offset_check.md:339-342`。
- 固定协议日志为 AP 43.0617 / AP50 72.8274 / AP75 45.1043 / APs 35.74 / APm 51.02 / APl 44.95；导出为 43.18 / 72.93 / 45.26 — `docs/fdr_variance_step0.md:584-586`、`:705-712`。

**Oracle 上限（均依赖 GT，只作诊断）**

| 变换 | 旧协议导出 AP/AP50/AP75/APs | 新协议导出 AP/AP50/AP75/APs | OGSOD（对照） |
|---|---|---|---|
| AP | 42.88 / 72.73 / 44.81 / 35.88 | 43.18 / 72.93 / 45.26 / 35.78 | 59.36 / 92.38 / 58.08 / 53.00 |
| AP_GTBOX | 72.72 / 72.73 / 72.72 / 67.48（+29.84） | 72.93 / 72.93 / 72.93 / 67.25（+29.75） | 92.37（+33.01） |
| AP_TPFP | 47.78 / 88.45 / 47.15 / 39.71（+4.90） | 47.84 / 87.95 / 47.39 / 39.86（+4.66） | 60.41（+1.05） |
| AP_IQS | 50.94 / 81.88 / 53.84 / 42.78（+8.06） | 51.20 / 81.98 / 54.20 / 43.09（+8.02） | 64.25（+4.89） |
| AP_BOTH | 88.45（+45.57） | 87.95（+44.77） | 96.70（+37.34） |

来源：`docs/sardet_offset_check.md:391-397`、`docs/fdr_variance_step0.md:710-716`、`docs/localization_diagnosis_2026-10-24.md:8-12`。

**变换的定义**
- AP_GTBOX 把 IoU≥0.5 匹配上的框换成 GT 框；AP_TPFP 把未匹配检测的分数置 0；AP_IQS 把分数乘以与同类 GT 的最佳 IoU；AP_BOTH 是 GT 框加未匹配置 0 — `tools/localization_ceiling.py:11-18`、`:122-132`。

**系统偏移**
- 全局 dx=+0.194px、dy=−0.185px（8,324 对）；OGSOD 式的统一 x 偏移没有复现；最佳平移的 oracle 增益只有约 +0.26 AP（OGSOD 为 +5.12）— `docs/sardet_offset_check.md:404-405`、`:442-444`、`:451-460`。
- 定位到并修复的是导出工具宽高用反的问题，生产管线未发现固定偏移 — `docs/sardet_offset_check.md:20-40`、`:482-484`。

**中心与尺寸误差**
- 匹配对平均 IoU 0.784。总体尺寸修正 +0.053、中心修正 +0.051（约 51%/49%）；<32² 档尺寸 +0.053、中心 +0.068（约 44%/56%）— `docs/sardet_offset_check.md:486-497`。
- 中心误差标准差（源像素）：harbor 7.9/6.3、aircraft 4.8/4.4、bridge 3.8/2.7、ship 2.6/2.5、car 1.9/2.3、tank 1.3/1.0 — `docs/sardet_offset_check.md:409-414`。

**FDR 方差（第 0 步）**
- 分布与框同源，recon_err=2.38×10⁻⁷ — `docs/fdr_variance_step0.md:630`。
- corr(σ, |边误差|) 四边中位数 0.430（6,031 对），通过 — `docs/fdr_variance_step0.md:638-646`。
- <32 像素档中位数 0.395，低于整体，按预设判定终止 — `docs/fdr_variance_step0.md:654-661`。
- 四边 max/min=1.0498，L/R=0.990，T/B=0.990，按预设判定终止，"SAR 上下边物理不确定性不同"不成立 — `docs/fdr_variance_step0.md:667-678`。
- 五分位校准四边全部单调，误差约 0.10→0.45 — `docs/fdr_variance_step0.md:686-693`。
- mean(σ) 与 IoU 的相关 −0.583，与分数的相关 −0.811，均只在匹配样本上计算 — `docs/fdr_variance_step0.md:696-698`。
- 结论是终止逐边各向异性和小目标强化路线，但不否定方差与误差的关系 — `docs/fdr_variance_step0.md:732-735`。
- 训练后的 LQE 末层已非零，所以 σ 信号与 LQE 的独立性未知 — `docs/fdr_variance_step0.md:622-624`。

**σ 的排序价值：未测量**
- `docs/sigma_ranking_value.md` 只有实验设计和判读标准（关键量 `delta(AP_both − AP_score)`，≥+1.5 有价值，±0.5 内关闭）— `docs/sigma_ranking_value.md:56-57`、`:421-426`。
- 文件要求回填结果（`:445-456`），但全文没有执行记录。
- 设计中要新建的 `tools/sigma_ranking_value.py`（`:86`）在当前仓库不存在（本次已检查 `tools/` 目录）。

**全集 baseline：只有协议，没有结果**
- `docs/sardet100k_full_baseline.md` 定义了配置、94,493/10,492 数据、2952 步/轮、合计 354,240 步和启动命令（`:3`、`:11-12`、`:30-33`），没有任何 AP 数字。
- `docs/` 下对该运行名的检索只命中这一份文档本身；8 份指定文档中都没有全集 AP。
- 子集协议文档明确说子集只用于方向筛选，方法效果必须在全集独立验证 — `docs/sardet_paper120_nozoom_nocrop.md:48`。

### Inferences
- **AP_GTBOX 的 +29.75 不是独立测量（推断）**：把匹配框换成 GT 后，所有 IoU 阈值的结果都等于 AP50，所以这个增量在机制上就是 AP50−AP。`AP_GTBOX≈AP50` 这一点在 `docs/localization_diagnosis_2026-10-24.md:21-22` 也被观察到。它可以和文献中的 AP50−AP 直接比较。
- **与文献的核心对应（推断）**：本地子集 AP50−AP=29.8，文献全量方法多为 29–32（Deformable DETR、AutoAssign 为 36–37）。AP50 却差很多（72.9 对 86–89）。也就是说，从 10% 数据到全量，AP50 变化显著，而定位造成的落差几乎不变。这与"多数据能减少漏检和误检，但对高 IoU 精度帮助有限"一致。注意模型、输入、val 都不同，这个比较只是方向性的。
- **本地损失瀑布（自行计算，新协议，路径依赖）**：43.18 → 72.93（定位 +29.75，占 56.82 的 52%）→ 87.95（误检/排序 +15.02，26%）→ 100（无候选 +12.05，21%）。换一种顺序数字会变。全量上定位占比预计会高（文献 67–74%）；子集诊断可能高估误检和漏检的份额。OGSOD 上"前后景已饱和"的否定不能迁移到 SARDet，但 SARDet 子集上"前后景空间大"也不能直接外推到全集。
- **AP_IQS 不是纯粹的排序质量上限（推断）**：没有同类 GT 重叠的检测 IoU=0，分数被乘成 0（`tools/localization_ceiling.py:111-116`、`:129-130`）。所以 +8 包含了抑制误检的效果，AP50 从 72.9 升到 82.0 就是证据。任何只看定位质量的后处理都拿不到全部 +8。
- **逐类召回上限（自行计算，旧协议）**：用匹配对数除以筛选 val 的 GT 数：ship 97.9%、aircraft 98.8%、car 99.4%、**tank 76.9%、harbor 79.7%、bridge 81.8%**（`docs/sardet_offset_check.md:409-414` 除以 `docs/sardet_screening_handoff.md:28-33`）。六类平均 88.9%，与 AP_BOTH 的 88.45 几乎相等。这说明 12 点左右的"无候选"损失几乎全部集中在 tank、harbor、bridge 三类，且其 IoU≥0.5 的同类框都没有出现。这个结果是新发现，但属于子集旧协议；导出只保留每个 query 的最大类别，会略低估召回。
- **与文献的一致点（推断）**：tank 和 bridge 在已发表表格里也是最低两类；bridge 还有最大的 AP75 落差。harbor 则不同：它在已发表结果中 AP 最高，而子集召回只有 79.7%，这可能是 10% 子集只有 134 张训练图的少样本效应（`docs/sardet_screening_handoff.md:33`）。
- **不要过度解读第 0 步（推断）**：`docs/sigma_ranking_value.md:18` 说各向异性"在 1 个标准误内不可区分"，但第 0 步记录明确说这个边界不是统计显著性检验（`docs/fdr_variance_step0.md:675-678`）。这一句在第 0 步记录里找不到依据。结论可以保留，理由应改成"差异太小，达不到预设阈值"。
- **偏移方向已关闭（推断）**：+0.26 AP 的 oracle 太小，不值得作为论文方向。

### Gaps
- 文档中没有子集或全集的**逐类 AP**（筛选文档只给了读取方法，`docs/sardet_screening_handoff.md:214-239`），本次无法把本地 bridge、tank 的 AP 与文献直接比较。上面的逐类结论来自匹配数，不是 AP。
- 单种子。子集协议的种子波动在所读文档中没有记录。
- 所读文档没有记录逐来源 AP，也没有全集的墙钟时间。
- 全集运行是否已在服务器上跑完未知：本次没有服务器访问，也只读了指定文档。

## 问题五：候选研究缺口排序，以及真实缺口与数据集人工产物的区分

### Takeaway
最值得先做的不是新模块，而是零训练的数据审计与逐来源评测：它证据最强、成本最低、别人没做，还决定后面其他缺口是否真实。其后是 bridge/tank 的高 IoU 损失（先用 RSAR 第二套标注判断是真实定位问题还是标注上限）和留一来源泛化。模块类方向（去噪、频域）证据弱且拥挤，不建议作为主线。

### Cited Findings
以下排序中引用的事实均已在前四节给出来源，这里只列排序依赖的关键几条：
- RSAR 明确指出重复和无标注图仍存在 — [RSAR §4.1](https://arxiv.org/pdf/2501.04440)。
- 没有可靠逐图传感器标签，是别人没做留一传感器评估的原因 — [MSPaDet §4.2.4](https://doi.org/10.3390/rs18101513)。
- 仓库有来源映射的线索 — [Issue #35](https://github.com/zcablii/SARDet_100K/issues/35)。服务器上已有 OGSOD 和 SAR-AIRcraft 的原始数据（`AGENTS.md` 的数据集路径约定；`docs/sar_aircraft_dataset.md:166`）。
- 已有相关性匹配工具：24×24 相关，阈值 0.9 — `tools/check_split_overlap.py`，用法见 `docs/sar_mechanism_survey_2026-11-03.md:197`。该工具目前针对 SAR-AIRcraft，用于 SARDet 需要小改。
- 本仓库已有多个模块实现后无收益的记录（SCAS、SBR、DPM、BCM、NWD、HWD）— `docs/fdr_variance_step0.md:8`。

### Inferences

**真实缺口与人工产物的判断（全部是推断，证据强度已标注）**

| 现象 | 判断 | 对 AP 的方向 | 证据强度 | 判定实验 |
|---|---|---|---|---|
| AP50−AP 约 30 点 | 大部分**真实**；上限未知，部分可能是标注噪声 | 压低 | 强（现象）/弱（归因） | E2 |
| bridge AP75 崩塌（.885→.368） | **未定**：可能是长条目标的真实定位困难，也可能是端点标注模糊 | 压低 | 强（现象） | E2 |
| tank/bridge/harbor 约 20% 无候选 | **待定**：可能真实漏检，也可能含切片碎片 GT | 压低 | 中（子集旧协议） | E1、E5 |
| 零面积、切片碎片 GT | **人工产物**：匹配不上，进入 val 就是必然漏检 | 压低 | 存在性强 / 数量未知 | E1 |
| 跨划分近似重复 | **人工产物** | 抬高 | 存在性强 / 比例只知 SAR-AIRcraft | E1 |
| car、harbor AP50 近乎饱和 | **很可能是人工产物**（单一来源的类别） | 抬高 | 中 | E3 |
| 多来源合并后的漏标 | **未定**；作者否认但未审计 | 压低（虚假 FP） | 弱 | E6 |
| APs 含义 | **评测人工产物**（混合尺度） | 不可比 | 中 | 按输入面积重算 |
| 榜单 52–69 的差距 | **协议人工产物** | 不可比 | 强 | 固定协议复跑 |
| 去噪模块的 +1~2 | 可能真实但低于训练长度效应 | 小 | 中 | 不优先 |

**候选排序**（可行性按两张 RTX 4090 评估；新颖性依据是本次检索未找到同类工作，不是证明不存在）

| # | 缺口 | 证据 | 可行性 | 新颖性 | 判定 |
|---|---|---|---|---|---|
| E1 | 重复、碎片 GT 和按见过/没见过的审计 | 强 | 极高，CPU 或少量 GPU，零训练 | 高 | **首选** |
| E2 | 高 IoU 损失的标注上限（RSAR 对照） | 强 | 高，CPU | 高 | **首选** |
| E3 | 逐来源 AP 与类别-来源混淆 | 中强 | 高，只需导出和评测 | 高 | **首选** |
| E4 | 留一来源泛化协议 | 中（别人明确说没做） | 中，子集上每折一次 120e | 高 | **主线候选** |
| E5 | tank/bridge 召回 | 中 | 中，需逐类诊断后才定方案 | 中 | 看 E1 |
| E6 | 漏标与假误检 | 弱到中 | 高，部分需人工目视 | 中高 | 看 E3 |
| E7 | σ 排序增益 | 弱（共线 −0.811） | 极高，CPU | 低（接近 GFL/LQE） | 便宜，可顺手关闭 |
| E8 | 斑点干净/加噪评测 | 中（只有模块，没有协议） | 高，零训练起步 | 中 | 备选 |
| E9 | 实时 DETR 全量基线 | 前提条件，不是缺口 | 354,240 步，约为子集 10 倍 | 中 | **必须先有** |

**前提 P0：全集 baseline。** 没有这个结果，上面所有全量判断都只是从子集外推。文献里最接近的对照是 RT-DETR 54.24（无预训练、无增强、150 轮）和 MDIS-DETR 58.82。全集 baseline 跑完后立即重跑 `tools/localization_ceiling.py` 和逐类 AP。

**各缺口的廉价证伪方式**

- **E1 审计（零训练）**：
  1. 只读 JSON，统计 val 中面积为 0、面积 <16 像素²、紧贴切片边界的 GT，按类别列出。
  2. 在现有检查点的导出上把这些 GT 设为 ignore，重算 AP。
  3. 用相关性匹配（不用哈希）在类别+尺寸分桶内找 val 目标在 train 中的最近邻，按 ≥0.9 分为见过和没见过两组，分别算 AP。
  - **证伪条件**：碎片 GT 占比 <1% 且忽略后 AP 变化 <0.3；近似重复率 <5% 且见过/没见过的 AP 差 <3 点。满足则这条线放弃。

- **E2 标注上限（零训练）**：把 RSAR 的旋转框转成水平外接框，按图和类别匹配 SARDet-100K 的水平框，统计每类 IoU 分布。RSAR 的 val 和 test 是独立手工标注的（[RSAR §4.1](https://arxiv.org/pdf/2501.04440)），可以当作一个"第二标注者"。
  - **判读**：若 bridge 的 IoU 中位数 <0.8 或低于 0.75 的占比很高，AP75 的相当部分是标注噪声，高 IoU 方向的上限应下调。若各类 IoU 都 >0.9，则 AP50−AP 主要是真实定位误差，定位方向成立。
  - **风险**：RSAR 是否沿用 SARDet 文件名未核实；RSAR 可能参考了原水平框，这样得到的是标注噪声的下限，而不是真实值。

- **E3 逐来源 AP（零训练）**：
  1. 取得来源标签：先试 `mapping.json`；再对 OGSOD 和 SAR-AIRcraft 做相关性匹配，car 归 SIVED，harbor 归 OGSOD，其余按图尺寸与类别组合推断。
  2. 用现有检查点算逐来源、逐类 AP。
  - **证伪条件**：同一类别不同来源的 AP 差 <5 点，且 bridge、tank 在 MSAR 与 OGSOD 之间也差异不大。这样来源不是重要因素，E4 的动机也减弱。
  - 部分来源（ship 涉及 6 个来源条目）标签可能无法完全恢复，需要说明覆盖率。

- **E4 留一来源（少量训练）**：先做一个最有区分度的折：把 SADD 或 SSDD 从 train 剔除，在同一个 val 上评估该来源的目标。在 10% 子集上跑一次固定 120e 协议，和基线成对比较。
  - **证伪条件**：被留出来源的 AP 下降 <5 点，说明模型不靠来源特定特征，这个方向不值得写成论文。
  - 下降很大时，才值得设计方法（如来源不变的正则）并在全量上验证。
  - 注意 OGSOD、SAR-AIRcraft 这样的大来源去掉后还要考虑对应类别是否只剩少数来源。

- **E5 召回**：在全集 baseline 上重算逐类匹配率。先排除 E1 的碎片 GT。
  - **证伪条件**：全量上 tank、bridge、harbor 的 IoU≥0.5 匹配率都 >92%，说明子集结果只是数据量不足，方向取消。

- **E6 漏标（零训练）**：取每类得分最高的 200 个 FP，按来源分组抽样目视（SAR-AIRcraft 上已有类似流程，`docs/sar_protocol_b.md:104`）。
  - **证伪条件**：疑似漏标的比例 <10%。满足就不做标注噪声方向。
  - 项目最近在 SAR-AIRcraft 上的结论也是疑似漏标解释的差距有限（提交 `d928709`），但这不能直接迁移到多来源合并的 SARDet。

- **E7 σ 排序（零训练）**：按 `docs/sigma_ranking_value.md` 创建缺失的脚本，用 score>0.001、top-300 的完整导出跑两折交叉拟合，只看 `delta(AP_both−AP_score)`。
  - **证伪条件**：|delta|<0.5，方向关闭（预设标准）。
  - 由于 σ 与分数相关 −0.811，又有 LQE 在，我的先验是冗余（推断），但花费很小，值得用它关闭这个悬而未决的问题。

- **E8 斑点（零训练起步）**：对 val 图重新采样斑点（按图估计 L，见 `docs/sar_mechanism_survey_2026-11-03.md:453`），不重训，只测 AP 变化和预测翻转率。
  - **证伪条件**：AP 下降 <1 点。说明现有模型已经不敏感，鲁棒性方向没有空间。
  - JPEG 与 8 位量化会影响模拟的真实性。

**建议的执行顺序（推断）**

1. 取得 P0 全集 baseline 结果，以及逐类 AP、`localization_ceiling` 结果。
2. 并行做 E1、E2、E3、E7，都不占训练 GPU，几天内可得出结论。
3. 根据结果决定主线：
   - 若审计结果显著（重复多、来源差异大）：论文主线可以是"SARDet-100K 的评测结构与跨来源泛化"，方法部分配 E4。
   - 若审计无显著问题，E2 又显示标注足够干净：主线转向 bridge、tank 的高 IoU 定位。
   - 若 E2 显示标注噪声是上限：高 IoU 定位不再是好方向，改做检出/误检（全量上再看 E5、E6）。

**可发表性的现实判断（推断）**
- 纯审计可以成为论文的分析贡献，但是否足以单独成文，取决于量化后的结果有多强。
- 用户对"官方划分和指标不变"的决定仍然成立：审计只作诊断或补充报告，正式对比继续用官方划分。对 SARDet-100K 公开见过/没见过结果是否合适，需要用户决定。
- 任何"超过 SOTA"的说法，都要说明对照的 val 或 test、输入尺寸和训练长度；MDIS-DETR 的 58.82 与 DenoDet V2 的 56.71 协议不同（640/150e 对 512/12e），不能直接当作同一把尺子。

### Gaps
- E1、E2、E3 的结果目前都未知，本表的排序依赖它们；任何一项结果都可能调整后续优先级。
- 所有候选都没有实际增益证据。上述证伪阈值是我提出的经验值，不是文献标准，开跑前应由用户确认。
- 本笔记没有运行任何训练、推理或数据分析，未修改除本文件外的任何仓库文件。
- `tools/check_split_overlap.py` 用于 SARDet 的改造量和耗时未评估。全集 baseline 的墙钟时间在所读文档中没有记录。
