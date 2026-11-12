# SARDet-100K 上的 query-based 检测器、SAR 预训练与基础模型（2025-2026 精读笔记）

阅读范围与可信度说明：
- **全文直读（DOM/表格逐行提取）**：MDIS-DETR、CGAQ-DETR、SARLite（Table 1）。
- **全文经抓取工具提取（arXiv HTML，表格数字由工具转录，个别表可能对齐有误）**：SARDet-100K/MSFA、DenoDet V2、SARATR-X、SARATR-X-v2、SAMBA、SARMAE、ViTP、BabelRS、Pu & Xu MAE、RSAR、SAR-NAS、SARES-DEIM、MaDiNet、Inkawhich。
- **公开配置/日志直读**：SARATR-X-v2 `results/detection/sardet/`、SARATR-X `hivit_base_SARDet.py`、SARDet_100K `deformable-detr_r50_IN_sup_sar.py`。
- **未验证（仅摘要或搜索摘要）**：MD-DETR（仅摘要，全文需登录）、SA-DETR（仅搜索摘要）、SUMMIT（ScienceDirect 被拒，仅摘要级信息与他文转引）、IRT-DETR（仅搜索摘要）、MSPaDet（仅搜索摘要）。
- 下文 “mAP” 均指 COCO AP@[.50:.95]，除非另注。

## 各方法的架构改动、训练协议与表内结果

### Takeaway
SARDet-100K 上已有 RT-DETR 系工作（MD-DETR、SA-DETR、MDIS-DETR），基础模型则从 57 涨到 68.7；但各文协议差异极大（7/12/36/150/500 epoch，512/640/800 输入，是否预训练、是否增强），表内数字多为跨协议拼接，不能直接排名。

### Cited Findings

**A. 原始基线：SARDet-100K / MSFA（NeurIPS 2024，参考）**
- 数据：10 源合并，116,598 图 / 245,653 实例，train/val/test = 94,493/10,492/11,613；原有划分沿用，否则 8:1:1；大于 1000×1000 的图按 512×512、重叠 200 像素切片（AIR_SARShip 1、MSAR、SAR-AIRcraft）（Table 1-2、App. A.2）— [SARDet-100K](https://arxiv.org/html/2403.06534v3)
- 协议（App. A.6, Table S12）：800×800，RandomFlip 0.5，AdamW，batch 16，lr 1e-4，12 epoch，取第 12 epoch 权重，8×RTX 3090；未报告 seed — [SARDet-100K](https://arxiv.org/html/2403.06534v3)
- 配置与论文不一致：公开配置 `deformable-detr_r50_IN_sup_sar.py` 写 `max_epochs = 50`、`lr=0.0002`、`num_queries=300`，预训练路径为 `.../r50_sar/epoch_100.pth`，与论文 “12 epoch、lr 1e-4” 不同 — [config](https://raw.githubusercontent.com/zcablii/SARDet_100K/main/MSFA/local_configs/SARDet/other_detectors/deformable-detr_r50_IN_sup_sar.py)
- 原始论文**没有 DINO、DINO-DETR、RT-DETR 行**；query-based 只有 DETR、Deformable DETR、DAB-DETR、Sparse R-CNN（Table S8）— [SARDet-100K](https://arxiv.org/html/2403.06534v3)；模型库配置同样只列这四个 — [repo](https://github.com/zcablii/SARDet_100K)

Table S8（R50，mAP/AP50/AP75/APs/APm/APl）— [SARDet-100K](https://arxiv.org/html/2403.06534v3)

| 检测器 | 预训练 | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|---|
| Faster R-CNN | ImageNet / MSFA | 49.0 / 51.1 | 82.2 / 83.9 | 52.9 / 54.7 | 43.5 / 45.2 | 60.6 / 62.3 | 55.0 / 57.5 |
| Cascade R-CNN | ImageNet / MSFA | 51.1 / 53.9 | 81.9 / 83.4 | 55.8 / 59.8 | 44.9 / 47.2 | 62.9 / 66.1 | 60.3 / 63.2 |
| GFL | ImageNet / MSFA | 49.8 / 53.7 | 80.9 / 84.2 | 53.3 / 57.8 | 42.3 / 47.8 | 62.4 / 66.2 | 58.1 / 59.5 |
| DETR | ImageNet / MSFA | 31.8 / 47.2 | 62.3 / 77.5 | 30.0 / 49.8 | 22.2 / 37.9 | 44.9 / 62.9 | 41.1 / 58.2 |
| Deformable DETR | ImageNet / MSFA | 50.0 / 51.3 | 85.1 / 85.3 | 51.7 / 54.0 | 44.0 / 44.9 | 65.1 / 65.6 | 61.2 / 61.7 |
| DAB-DETR | ImageNet / MSFA | 45.9 / 48.2 | 79.0 / 81.1 | 47.9 / 51.0 | 38.0 / 41.2 | 61.1 / 63.1 | 55.0 / 55.4 |
| Sparse R-CNN | ImageNet / MSFA | 38.1 / 41.4 | 68.8 / 74.1 | 38.8 / 41.8 | 29.0 / 33.6 | 51.3 / 53.9 | 48.7 / 53.4 |

- 最佳主干（Table S9，Faster R-CNN）：ConvNeXt-B 88.6M，ImageNet 55.1 → MSFA 56.4（AP50 88.2，AP75 61.5，APs/m/l 51.1/68.3/62.4）；ConvNeXt-T 53.2 → 54.8 — [SARDet-100K](https://arxiv.org/html/2403.06534v3)
- 分类别（Table S13，ConvNeXt-B+MSFA，mAP）：Ship 66.9，Aircraft 45.5，Car 65.5，Tank 45.4，Bridge 44.1，Harbor 71.1 — [SARDet-100K](https://arxiv.org/html/2403.06534v3)

**B. 2025-2026 RT-DETR / DETR 系**

*MDIS-DETR（Remote Sensing, 2026-06，全文）* — [MDPI](https://www.mdpi.com/2072-4292/18/13/2076)
- 改动位置：输入层 LSFDM（原图、Gaussian、bilateral 三路可学习加权+残差，§3.2）；encoder 内加入 PAFDM，针对 AIFI 自注意力的频域问题（MS-PLA 极性注意力 + AFDFN 窗口 FFT 可学习滤波，§3.3）；跨尺度融合阶段用 NICFM 做非顺序交互（§3.4）。**论文未描述对 decoder 与 query 选择的改动**。
- 协议（§4.2）：RTX 4090 单卡，**ResNet-18、backbone 从零训练、无预训练、无显式数据增强**，batch 16，AdamW，lr 1e-3，SARDet-100K 150 epoch，其余沿用 RT-DETR 默认；未报告 seed。输入尺寸公式在网页中被 MathJax 吞掉，未读到。
- Table 3 消融（FLOPs/Params/FPS/mAP/AP50/AP75/APs/APm/APl）：

| 配置 | FLOPs | Params | FPS | mAP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|---|---|---|
| RT-DETR 基线 | 56.97 | 19.88 | 187.93 | 54.24 | 85.89 | 58.41 | 48.74 | 67.72 | 62.54 |
| +空间域 | 56.97 | 19.88 | 141.80 | 56.21 | 87.52 | 60.91 | 50.83 | 69.04 | 64.32 |
| +频域 | 57.21 | 20.05 | 198.87 | 56.25 | 87.53 | 60.82 | 51.02 | 69.29 | 64.23 |
| +结构域 | 66.09 | 22.24 | 145.21 | 57.32 | 88.28 | 62.03 | 52.11 | 70.40 | 64.39 |
| MDIS-DETR | 66.39 | 22.42 | 150.17 | 58.82 | 89.31 | 63.68 | 53.43 | 71.37 | 67.08 |

- Table 8 分类别 AP：RT-DETR Ship 65.94 / Aircraft 48.16 / Car 62.61 / Tank 43.33 / Bridge 38.79 / Harbor 66.63；MDIS-DETR 68.69 / 53.63 / 67.02 / 50.48 / 42.62 / 70.47。同表 GFL Aircraft 57.63，DenoDet Aircraft 57.36，高于 MDIS-DETR。
- 同表 FPS（作者给出，硬件/批量定义未细述）：RT-DETR 187.93，MD-DETR 105.33，CGAQ-DETR 76.52，GFL 36.41，Cascade R-CNN 14.54。

*MD-DETR（IEEE TGRS 2024，仅摘要，未验证）* — [IEEE](https://ieeexplore.ieee.org/document/10741545)
- 基于 RT-DETR：图像级（原图与去噪图拼接）、特征级（coarse-mask 引导 decoder 全局特征）、query 级（one-to-many 辅助头，动态调 query 数）。摘要报告 SARDet-100K AP50 88.5。
- 转引：MDIS-DETR Table 8 列 MD-DETR mAP 55.38 / AP50 87.96 / AP75 59.08，FLOPs 65.32，Params 43.91，FPS 105.33，与摘要 88.5 不一致 — [MDPI](https://www.mdpi.com/2072-4292/18/13/2076)

*SA-DETR（IEEE，仅搜索摘要，未验证）*：基于 RT-DETR，用 SAM 生成的掩膜做 encoder 去噪、mask-guided query selection 和实例分割；搜索摘要报告 SSDD AP50 99.0、SARDet-100K mAP50 88.4，全文未读到 — [IEEE](https://ieeexplore.ieee.org/document/10720452/)

*CGAQ-DETR（Remote Sensing, 2025-09，全文）* — [MDPI](https://www.mdpi.com/2072-4292/17/18/3254)
- 基于 Deformable DETR：CMFE（角点引导的多尺度特征增强）+ AQR（用底层特征回归目标数，动态调整 K/object query 数）；R50，300 queries，5 层 encoder/decoder，640×640，Adam，lr 1e-4，batch 4（§4.2）。
- **SARDet-100K 全部方法只训 7 epoch**（§4.2）。Table 4：CGAQ-DETR AP50 69.8 / AP75 39.7 / mAP 38.6 / APl 40.6 / APm 49.1 / APs 31.8；Deformable DETR 66.6/30.0/33.3；DETR 22.0/2.3/7.1；DAB-DETR 57.1/24.8/28.1；GFL 63.7/33.0/34.1；Cascade 65.6/34.0/35.9。
- 消融（Table 6）只在 FAIR-CSAR 上做（6 epoch），不在 SARDet-100K。

*SARES-DEIM（arXiv 2604.04127v2，全文前 10 万字符，但**未用 SARDet-100K**）* — [v2](https://arxiv.org/html/2604.04127)
- 在 DEIM-S 骨干深层加稀疏 MoE（SARESMoE）、neck 加 SPD 式 SDEP；640×640，300 epoch，batch 8，lr 8e-4（backbone 4e-4），Mosaic/Mixup 在 epoch 288 后关闭，单 A40；单次实验 seed 0；预训练未说明。
- HRSID（v2 Table I，mAP/AP50/Params/FLOPs/FPS，* 为作者复现）：DEIM-S* 73.0/92.0/10.2/24.8/237.9，DEIM-M* 75.6/93.1/19.2/56.3/143.4，D-FINE-S* 70.2/88.2/10.2/24.8/227.4，D-FINE-M* 75.1/91.0/19.2/56.3/143.8，RT-DETR-L* 59.1/87.1/28.8/104.8/106.4，YOLOv8-S* 61.7/88.2，YOLOv11-S* 61.6/87.9，SAR-D-FINE（转引）72.0/90.8/11.8/31.0/109.0，SARES-DEIM 76.4/93.8/90.8/72.6/136.5。
- SAR-Ship（v2 Table II）：DEIM-S* 71.3/97.6，D-FINE-S* 71.1/96.8，RT-DETR-L* 67.3/96.8，YOLOv11-S* 66.4/97.0，SARES-DEIM 71.7/98.1。
- 版本不一致：v1 Table I 的 RT-DETR 67.3/87.9、YOLOv8 69.0/93.0、YOLOv11 67.7/92.4 — [v1](https://arxiv.org/html/2604.04127v1)，与 v2 相差 7-8 点，引用时必须标版本。

**C. 扩散与 Mamba**
- MaDiNet（TCSVT 2025）：**未报告 SARDet-100K 结果**（仅图注和参考文献提及），用 SSDD、SAR-Ship、SAR-AIRcraft（Table 2-5）；DiffusionDet 式由噪声框反推真框，MambaSAR 为 3 层 Mamba + 3 层 agent attention（放在 Res4），无学习型 query。Table 5 SAR-AIRcraft mAP50 90.5。正文写高斯采样，未提 Gamma — [arXiv](https://arxiv.org/html/2411.07500)（搜索摘要称其引入 Gamma 分布 — [GitHub 搜索结果](https://github.com/JoyeZLearning/MaDiNet)，与所读 HTML 版不一致，存疑）
- 未发现扩散检测器在 SARDet-100K 上的全文可证结果；DiffDet4SAR 只见 SAR-AIRCRAFT-1.0 mAP50 88.4 — [arXiv](https://arxiv.org/pdf/2404.03595)（未精读，PDF 未解析）

**D. SAR 预训练 / 基础模型（水平框 SARDet-100K）**

SARATR-X-v2 Table X 汇总（各行来自原文，非统一复跑）— [v2](https://arxiv.org/html/2607.23238)

| 方法 | 主干 | mAP | AP50 | AP75 |
|---|---|---|---|---|
| MSFA | ConvNeXt-B | 56.4 | 88.2 | 61.5 |
| SARATR-X | HiViT-B | 57.2 | 88.5 | 62.6 |
| SUMMIT | ViT-B | 57.0 | 89.9 | 62.7 |
| SARMAE | ViT-L | 63.1 | – | – |
| ViTP | ViT-L | 59.7 | – | – |
| MaRS | SwinV2-B | 55.4 | – | 61.4 |
| SAMBA | Mamba | 59.7 | 89.1 | 68.0 |
| BabelRS | ViT-L | 63.3 | 91.7 | – |
| SARATR-X-v2 | iTPN-B | 63.4 | 92.5 | 70.1 |
| SARATR-X-v2 | iTPN-L | 68.7 | 93.9 | 75.3 |

- **SARATR-X**（IEEE TIP 2025）：HiViT（66M，4×4 小 patch），目标改为多尺度比值梯度 MGF（r=9,13,17），ImageNet MIM → SAR MIM 两步，预训练 200 epoch、batch 800（§III-C，Table IX）。Table VII 57.3/88.7/62.8，Table XII 57.2/88.5/62.6（APs/m/l 51.2/69.6/64.7），两表不一致 — [SARATR-X](https://arxiv.org/html/2405.09365v4)
- SARATR-X 公开配置：**Faster R-CNN + FPN，36 epoch**，训练多尺度 (480–800, 1333)、AutoAugment 含 RandomCrop，AdamW lr 5e-4，layer decay 0.9，val/test 均为 `test.json`，`save_best='auto'`，未设 seed — [config](https://raw.githubusercontent.com/waterdisappear/SARATR-X/main/detection/configs/_hivit_/hivit_base_SARDet.py)。
- **SARATR-X-v2**（arXiv 2026-07，未标会议/期刊）：iTPN-B/L + MAE 式解码，目标为 1 个盲点聚合 + 5 个方向 log-ratio（r=3,5,9,13,17）softmax 融合，224 输入，掩码 0.75，1200 epoch，约 320K 无标签图（§III-B/D，§IV-A）— [v2](https://arxiv.org/html/2607.23238)
- v2 公开配置（`pixel_itpn_GFL_sardet100k.py`）：**GFL，stacked_convs=6，36 epoch**，训练多尺度 384/512/640，测试 512，6×4 GPU=24 batch，AdamW lr 2e-4，backbone lr_mult 0.1，layer decay 0.9，fp16，seed 42，deterministic — [config](https://raw.githubusercontent.com/Pepu0309/SARATR-X-v2/main/results/detection/sardet/pixel_itpn_GFL_sardet100k.py)；L 版 layer decay 0.95，其余相同 — [config](https://raw.githubusercontent.com/Pepu0309/SARATR-X-v2/main/results/detection/sardet/large_pixel_itpn_GFL_large.py)
- v2 日志：iTPN-L 第 36 epoch 0.687/0.939/0.753，APs/m/l 0.675/0.772/0.715，第 35 epoch 略低，与 Table X 对应最后 epoch — [log](https://raw.githubusercontent.com/Pepu0309/SARATR-X-v2/main/results/detection/sardet/large_20260421_033049.log.json)
- **SAMBA**（arXiv 2026-07）：双向 Mamba + 散射引导掩码 SG-MAE + SpatialMix 解码；约 27M、约 8 GFLOPs；Table IV 59.7/89.1/68.0；未写明检测头、协议、重复次数 — [SAMBA](https://arxiv.org/html/2606.31668)
- **SARMAE**（arXiv 2512.16635）：SAR-1M 共 1,312,902 张 SAR；SARE 噪声感知 MAE（50% 迭代注入加性高斯/乘性瑞利/Gamma 多视/均匀噪声，§8）；**Faster R-CNN + FPN，ViT 除末层外全部冻结**，batch 16，AdamW lr 1e-4（§12.2，Table 12）；Table 3 ViT-B 57.90、ViT-L 63.10 — [SARMAE](https://arxiv.org/html/2512.16635)
- **ViTP**（arXiv 2509.17562）：把 ViT 放进 VLM 用视觉指令微调；输入 800×800，Table 2 59.7；预训练集含 ISPRS_SAR、SAR_Sentinel — [ViTP](https://arxiv.org/html/2509.17562v3)；项目 README 列 Cascade-RCNN 57.9，与论文 59.7 不一致且文内未解释 — [README](https://huggingface.co/GreatBird/ViTP/blob/main/README.md)（仅搜索摘要）
- **BabelRS**（arXiv 2603.01758）：SARLang VQA 语言枢轴预训练，InternVL 初始化；Table 2 63.30（AP50 91.70），SM3Det 60.64；Table 3 同协议 ViT-L 初始化：MAE 39.48、DINOv2 41.38、CLIP 42.00、BEiT 50.10、SatMAE 47.90；检测头、输入尺寸、epoch 未写明；微调 lr 5e-5、batch 4×8 GPU — [BabelRS](https://arxiv.org/html/2603.01758)
- **Pu & Xu MAE**（arXiv 2501.11249，Fudan）：ViTDet ViT-B，SARDet-100k **训练集**无标注 MAE 400 epoch，512×512 微调 12 epoch；验证集 mAP：从零 47.2、ImageNet 53.6、SAR MAE 54.9（Table I）— [arXiv](https://arxiv.org/html/2501.11249v1)
- **SUMMIT**（JAG 2025，未验证全文）：ViT，MIM + 自监督去噪 + 散射特征增强；MuSID 超 56 万图，448 输入 — [HIT 摘要页](https://scholar.hit.edu.cn/en/publications/summit-a-sar-foundation-model-with-multiple-auxiliary-tasks-enhan/)、[HF README](https://huggingface.co/DYunt/SUMMIT/blob/main/README.md)（后者仅搜索摘要）；57.0/89.9/62.7 为 SAMBA Table IV 与 v2 Table X 转引 — [SAMBA](https://arxiv.org/html/2606.31668)
- **SAR-JEPA**：仓库只有分类/few-shot 任务（MSTAR、FUSAR-Ship、SAR-ACD），无检测配置、无 SARDet-100K 结果 — [repo](https://github.com/waterdisappear/SAR-JEPA)
- **Inkawhich**（arXiv 2509.21722）：不含 SARDet-100K，无检测，仅切片分类 — [arXiv](https://arxiv.org/html/2509.21722v1)

**E. 与 SARDet-100K 的关系：RSAR（CVPR 2025）**
- RSAR 由 SARDet-100K 构建，H2RBox-v2 + UCR 生成伪旋转框后人工校正，val/test 全人工标注（§4）；95,842 图 / 183,534 实例，6 类 — [RSAR](https://arxiv.org/html/2501.04440v1)；README 划分 78,837/8,467/8,538 — [gitee](https://gitee.com/buptybx/RSAR)（搜索摘要）。
- 原文：“Although SARDet-100K has processed the data, duplicate and unannotated images remain.” 清洗后对重复图只保留标注最多的一张（§4.1）— [RSAR](https://arxiv.org/html/2501.04440v1)
- Table 4（OBB，MMRotate 默认）：RoI-Transformer 35.02/66.95/32.65 最高；Deformable DETR 19.63/46.62/13.06；ARS-DETR 31.56/61.14/28.97 — [RSAR](https://arxiv.org/html/2501.04440v1)

### Inferences
- 用 RSAR 与 v2 的 rotated 对比时要分清任务：SARDet-100K 的 HBB 协议数字不能与 RSAR 的 OBB 协议混排。
- MDIS-DETR 未描述 query 改动，也就是说当前已知最完整的 RT-DETR 系工作把增益放在输入/encoder/跨尺度融合；SARDet-100K 上针对 D-FINE/DEIM 回归头或密集匹配的改动，至少未见全文可证的工作。

### Gaps
- MD-DETR、SA-DETR 全文（表格、协议、RT-DETR 基线）未读到；SUMMIT 全文（Table 2、MuSID 组成）未读到（ScienceDirect 被拒）。
- MDIS-DETR 输入尺寸未能从网页恢复。
- SARMAE、ViTP、BabelRS 的 SARDet-100K 分类别和 APs/m/l 未见报告。

## DETR 家族相对 CNN 的排名、原因，以及 real-time DETR 是否真是缺口

### Takeaway
“real-time DETR on SARDet-100K” 不是整体空白：RT-DETR 已有结果（MDIS-DETR Table 3 给出 R18 基线 54.24）。在我检索范围内，**未发现** D-FINE、DEIM、DEIMv2、RF-DETR、LW-DETR 在 SARDet-100K 上的报告，但这只是未检索到，不是不存在的证明。更窄的真实缺口是：现代实时 DETR 用标准 COCO 初始化、标准增强、多 seed，并与长训 YOLO 公平对比。

### Cited Findings
- 检索词（WebSearch，均无 D-FINE/DEIM/DEIMv2/RF-DETR/LW-DETR 的 SARDet-100K 结果）：`RT-DETR SARDet-100K real-time SAR object detection`；`D-FINE SAR ship aircraft detection SARDet-100K DEIM`；`RF-DETR LW-DETR DEIMv2 SAR image detection SARDet-100K`；`"SARDet-100K" "D-FINE" OR "DEIM" OR "RF-DETR" comparison table SAR detector`；`DEIMv2 SAR object detection improved DEIM synthetic aperture radar 2026`；`"SARDet-100K" real-time transformer detector lightweight RT-DETR improved mAP FPS 2026`；`"SARDet-100K" DETR query-based detector SAR 2026 denoising`；`SAR-D-FINE SAR ship detection D-FINE improvement HRSID 2025`；以及任务指定的原样短词 `D-FINE SAR`（无 SAR 相关命中）、`DEIM SAR`（只命中 SARES-DEIM）、`RT-DETR SARDet-100K`（搜索本身未直接命中，RT-DETR 数字来自 MDIS-DETR 全文）、`real-time SAR detection SARDet-100K`（命中 SARLite、SAR-NAS、CGAQ-DETR）。检索仅用 WebSearch（美国地区），未直接查 CNKI、IEEE/Springer 站内检索、Google Scholar 引用链，可能遗漏。
- D-FINE/DEIM 在 SAR 上最接近的结果是 SARES-DEIM，只用 HRSID 和 SAR-Ship；其 v2 表中 DEIM-S 在 HRSID 上 73.0，高于 YOLOv11-S 61.6 和 RT-DETR-L 59.1 — [arXiv v2](https://arxiv.org/html/2604.04127)；其中 SAR-D-FINE 只见该表转引，独立检索未找到原文。
- IRT-DETR（J. Supercomputing, 2026-01）搜索摘要称在 SARDet-100K、MSAR-1.0、SAR-AIRcraft-1.0 上评估，但全文和数字未读到，标记未验证 — [ResearchGate](https://www.researchgate.net/publication/399890033_IRT-DETR_a_lightweight_transformer_with_ConvNextV2_and_Cascaded_Group_Attention_for_real-time_military_aircraft_detection_in_aerial_imagery)
- MDIS-DETR 原文写 “the performance of RT-DETR in complex SAR imagery has not been systematically studied”（§2.1）— [MDPI](https://www.mdpi.com/2072-4292/18/13/2076)

**同一协议内的 DETR vs CNN**
- DenoDet V2 Table S7（512×512，12 epoch，batch 16，DAdaptAdam，4×RTX 4090，IN 预训练，未报 seed）28 个方法中：Deformable DETR 52.00 排第 17；DETR 45.73 第 23；Conditional DETR 44.04；DAB-DETR 43.31。前列为 DenoDet V2 56.71、DenoDet 55.88、GFL 55.01、ATSS 54.95、TOOD 54.65 — [DenoDet V2](https://arxiv.org/html/2508.09392)
- 同表 Deformable DETR AP50 88.77（仅次于 AutoAssign 89.58）但 AP75 只有 54.03；GFL AP50 85.16、AP75 58.87 — [DenoDet V2](https://arxiv.org/html/2508.09392)
- MSFA Table S8（ImageNet）10 个中 Deformable DETR 50.0 排第 2（仅次于 Cascade 51.1），MSFA 后降到第 4 — [SARDet-100K](https://arxiv.org/html/2403.06534v3)
- Sparse R-CNN 38.1/41.4，为 query-based 最差之一 — [SARDet-100K](https://arxiv.org/html/2403.06534v3)
- RSAR OBB：Deformable DETR 19.63 垫底，ARS-DETR 31.56 仍低于 RoI-Transformer 35.02 — [RSAR](https://arxiv.org/html/2501.04440v1)

**实时检测器的对手水平**
- SAR-NAS（arXiv 2509.01279，未标会议）Table 1-2：YOLOv10-N 59.47（2.27M/6.53G），YOLOv10-S 68.84（7.22M/21.43G），SAR-NAS-N 60.18（1.97M/5.96G），SAR-NAS-S 69.27（7.07M/20.17G）；表内没有 RT-DETR；输入尺寸、预训练、基线设置未报告；基线行与 DenoDet V2 S7 数值相同 — [SAR-NAS](https://arxiv.org/html/2509.01279)
- SARLite（Sci. Rep. 2026）：640×640，SGD lr 0.02，batch 64，200 epoch，仅水平翻转，两张 RTX 4090；Table 1 YOLOv8-s 63.7/90.3，YOLO-World 62.7/89.8，YOLOv10-s 65.1/90.9，YOLO11-s 64.0/91.2，SARLite 67.7/92.8（7.8M/20.3G）；表内没有 RT-DETR — [SARLite Table 1](https://www.nature.com/articles/s41598-026-49143-5/tables/1)，协议见 [全文](https://www.nature.com/articles/s41598-026-49143-5)

### Inferences
- **排名高度依赖协议**：Deformable DETR 在 MSFA 表第 2、在 DenoDet V2 表第 17，所以不宜说 “DETR 系天然弱于 CNN”。
- 可能原因（推断，未有论文专门隔离验证）：一是 12 epoch 短训练不利于 DETR 收敛，DETR 从 31.8（MSFA 协议）到 45.73（DenoDet 协议）的波动也支持这点；二是 Deformable DETR 高 AP50、低 AP75，说明定位精度是短板，这恰好是 D-FINE/DEIM 针对的方向，但在 SARDet-100K 上未被验证。
- MDIS-DETR 的 RT-DETR 54.24 是**从零训练且无增强**的 R18 结果，接近或略低于 12 epoch GFL（55.01），低于长训 YOLO（64-69），不代表实时 DETR 上限；RT-DETR Aircraft 48.16 也偏低。
- 对 DEIM/D-FINE 论文定位的含义：若只和 Deformable DETR 50-52 或 RT-DETR 54 比，会高估优势。合理对标是 YOLO11-s 64.0、YOLOv10-S 65.1–68.84、SAR-NAS-S 69.27、SARATR-X-v2 iTPN-L 68.7，且应同协议复跑。

### Gaps
- 无法确认 CNKI、付费期刊或未收录的预印本中没有 D-FINE/DEIM on SARDet-100K。
- MD-DETR/SA-DETR 的 RT-DETR 基线协议未知，无法判断 MDIS-DETR 的 54.24 是否代表常规 RT-DETR 水平。
- 两篇 YOLO 文章 YOLOv10-s 相差 3.74（65.1 vs 68.84），原因未读到（SAR-NAS 未报输入尺寸和设置）。

## 性能来自预训练还是检测器设计，以及预训练泄漏风险

### Takeaway
SAR 专用预训练相对 ImageNet，在**同一协议**下的增益通常是 1-4 点（原版 DETR 除外），模块增益约 1-5 点；而协议差（epoch、输入、优化器）可达 10-14 点，主干放大约 5-6 点。68.7 这类高分不能全归于 SAR 预训练。泄漏方面，SARATR-X 系预训练集包含 SARDet-100K 全部 10 个来源，且 RSAR 证实 SARDet-100K 存在重复图，训练/测试隔离未被独立验证。

### Cited Findings

**增益对比**

| 类型 | 比较 | 增益 | 来源 |
|---|---|---|---|
| 预训练 | 从零 → ImageNet（ViTDet ViT-B） | +6.4 | [Pu & Xu Table I](https://arxiv.org/html/2501.11249v1) |
| 预训练 | ImageNet → SAR MAE（同上） | +1.3 | [Pu & Xu Table I](https://arxiv.org/html/2501.11249v1) |
| 预训练 | MSFA：Deformable DETR / Faster / Cascade / GFL | +1.3 / +2.1 / +2.8 / +3.9 | [Table S8](https://arxiv.org/html/2403.06534v3) |
| 预训练 | MSFA：DETR（异常值） | +15.4 | [Table S8](https://arxiv.org/html/2403.06534v3) |
| 预训练 | MSFA：ConvNeXt-B | +1.3 | [Table S9](https://arxiv.org/html/2403.06534v3) |
| 主干 | R50 → ConvNeXt-B（ImageNet，Faster） | +6.1 | [Table S9](https://arxiv.org/html/2403.06534v3) |
| 主干 | SARMAE ViT-B → ViT-L | +5.2 | [SARMAE Table 3](https://arxiv.org/html/2512.16635) |
| 主干 | iTPN-B → iTPN-L | +5.3 | [v2 Table X](https://arxiv.org/html/2607.23238) |
| 模块 | GFL → DenoDet V2 | +1.7 | [DenoDet V2](https://arxiv.org/html/2508.09392) |
| 模块 | DenoDet V1 → V2 | +0.83 | [DenoDet V2 S7](https://arxiv.org/html/2508.09392) |
| 模块 | RT-DETR → MDIS-DETR（单模块 +1.97/+2.01/+3.08） | +4.58 | [MDIS Table 3](https://www.mdpi.com/2072-4292/18/13/2076) |
| 协议差 | Faster R-CNN R50 IN：39.22 vs 49.0 | 9.78 | [DenoDet V2](https://arxiv.org/html/2508.09392)、[MSFA](https://arxiv.org/html/2403.06534v3) |
| 协议差 | DETR R50：31.8 vs 45.73 | 13.93 | 同上 |

- 自然图像 ViT 预训练在 BabelRS 协议下分数很低（MAE 39.48，DINOv2 41.38，BEiT 50.10，均 ViT-L），但这是 BabelRS 自家设置，头与 epoch 未写明 — [BabelRS Table 3](https://arxiv.org/html/2603.01758)
- SAMBA 称 Mamba 比 Transformer 参数和计算量降 65% 以上，但未在 SARDet-100K 上给出检测模型 FPS — [SAMBA](https://arxiv.org/html/2606.31668)

**基础模型的非预训练差异（配置证据）**
- MSFA 协议 12 epoch（论文），SARATR-X 配置 36 epoch 多尺度增强，v2 配置 36 epoch 多尺度；检测头分别是 Faster R-CNN、Faster R-CNN、GFL；输入 800 / 800 / 训练 384-640、测试 512 — [MSFA](https://arxiv.org/html/2403.06534v3)、[SARATR-X config](https://raw.githubusercontent.com/waterdisappear/SARATR-X/main/detection/configs/_hivit_/hivit_base_SARDet.py)、[v2 config](https://raw.githubusercontent.com/Pepu0309/SARATR-X-v2/main/results/detection/sardet/pixel_itpn_GFL_sardet100k.py)
- v2 用 iTPN-L 68.7 称比此前最好结果 +5.4，但比较对象包括 Faster R-CNN 路线、冻结 ViT（SARMAE）、未统一协议 — [v2 Table X/§IV-A](https://arxiv.org/html/2607.23238)、[SARMAE §12.2](https://arxiv.org/html/2512.16635)
- 轻量 YOLO 不用 SAR 基础模型也达到相近区间：YOLOv10-S 68.84、SAR-NAS-S 69.27（7M 级）— [SAR-NAS](https://arxiv.org/html/2509.01279)，YOLO11-s 64.0 — [SARLite](https://www.nature.com/articles/s41598-026-49143-5/tables/1)

**泄漏证据**
- SARATR-X 14 个预训练源含 SIVED 1,044、SAR-Ship 39,729、AIR-SARShip 801、HRSID 5,604、SSDD 1,160、SADD 883、SAR-AIRcraft 18,818、MSAR 28,499、OGSOD 18,331 — [README](https://github.com/waterdisappear/SARATR-X)；这覆盖 SARDet-100K 全部 10 个来源（AIR_SARShip 1/2 合计 801）— [SARDet-100K Table 1](https://arxiv.org/html/2403.06534v3)
- SARATR-X 脚注写 “We removed the test sets of downstream tasks from the pre-training set samples”，但未说明 SARDet-100K 如何操作 — [SARATR-X](https://arxiv.org/html/2405.09365v4)
- v2 Appendix B 写预训练集 “test-excluded with respect to the downstream benchmarks”，测试划分 “in full at the image/chip level” 移除；其 17 个源含 MSAR、OGSOD、SAR-Ship、HRSID、SSDD、AIR-SARShip、SAR-AIRcraft（15,899）、SADD、SIVED — [v2](https://arxiv.org/html/2607.23238)
- SARMAE 的 SAR-1M Table 6 直接列入 SAR-Det100k 94,493 张（数字等于官方 train 划分），另含 SAR-AIRcraft 18,818，并说明 RSAR 与 SAR-Det100k 图像相同；称对下游测试集做了 hash 去重，但未说明是否覆盖 SARDet-100K 重叠 — [SARMAE](https://arxiv.org/html/2512.16635)
- Pu & Xu 明确只用 SARDet-100k 训练集做 MAE — [arXiv](https://arxiv.org/html/2501.11249v1)
- BabelRS 未把 SARDet-100K 列入预训练表，但 Appendix D 有示例标注 “SARLang and SARDet-100K”，是否使用其图像未说明 — [BabelRS](https://arxiv.org/html/2603.01758)
- RSAR：SARDet-100K 仍有重复图，清洗 “helps prevent evaluation bias caused by dataset leaks”（§4.1）— [RSAR](https://arxiv.org/html/2501.04440v1)
- 同行对比：Inkawhich 对含测试重叠数据的预训练模型自称可能是 “major violation”，改以 transductive 框架解释，且该模型最优主要因此 — [arXiv](https://arxiv.org/html/2509.21722v1)（分类任务，非 SARDet-100K，仅作方法参照）

### Inferences
- SARATR-X README 的各源数量基本等于数据集全量（如 HRSID 5,604、OGSOD 18,331、SAR-Ship 39,729），若真已移除测试集，表内数应更小；因此要么表列移除前数量，要么移除未生效，**无法从公开材料判定**。这不是泄漏的证明。
- 即便按 SARDet-100K test.json 移除图像/切片，仍可能残留：① SARDet-100K 自身重复图（RSAR 证实存在）；② 200 像素重叠切片（SAR-AIRcraft、MSAR、AIR_SARShip 1）在 train/test 间的邻接区域；③ 同一景源的相邻切片。对 train-only 预训练（SARMAE、Pu & Xu）同样可能成立。上述 ②③ 是推断，未见任一论文量化。
- 对性能归因：SARATR-X（57.3）相对 ConvNeXt-B+MSFA（56.4）只多 0.9，且配置为 36 epoch 对 12 epoch，**SAR 预训练净增益可能更小**；v2 的 63.4 相对 57.2 同时换了头、尺度、预训练目标和主干，单因素增益不可分离。
- 若本项目用 SAR 预训练，建议显式记录排除清单，并以 SARDet-100K 官方 train 为边界；SAR-AIRcraft 预训练到 SARDet-100K 评估属于高风险组合（SARDet-100K 含 SAR-AIRcraft 切片）。

### Gaps
- 未见任何论文公布 SARDet-100K 训练/测试之间的重复率，或发布排除清单；本笔记未自行计算。
- MuSID（SUMMIT）组成未读到，泄漏无法评估。
- 没有在同一协议下隔离 “头 / schedule / 预训练” 三因素的论文。

## 弱点：单 seed、计算量、非实时、单检测器增益、数字不一致

### Takeaway
基础模型结果几乎全是单 seed、高计算量、未报告 FPS；RT-DETR 系有 FPS 但基线训练条件极不典型；跨论文表格大量转抄且存在错误，SOTA 宣称需谨慎。

### Cited Findings

**单 seed / 无方差**
- SARATR-X-v2 配置 seed 42；论文仅在 ATRNet-STAR 线性探针中 “averaged over three random seeds”，检测无方差 — [v2](https://arxiv.org/html/2607.23238)、[config](https://raw.githubusercontent.com/Pepu0309/SARATR-X-v2/main/results/detection/sardet/pixel_itpn_GFL_sardet100k.py)
- SAMBA Section IV 未报告重复、seed 或方差 — [SAMBA](https://arxiv.org/html/2606.31668)；SARES-DEIM v2 单次实验用 seed 0，仅路由鲁棒性研究加 seed 13/21/42 — [arXiv v2](https://arxiv.org/html/2604.04127)；DenoDet V2、MDIS-DETR 未报 seed — [DenoDet V2](https://arxiv.org/html/2508.09392)、[MDPI](https://www.mdpi.com/2072-4292/18/13/2076)
- 例外：SARLite 对消融做 5 折交叉验证和配对 t 检验（但 5 折划分不是官方 split）— [SARLite](https://www.nature.com/articles/s41598-026-49143-5)

**验证集用法**
- SARATR-X 和 v2 配置都把 val 指向 `Annotations/test.json`，并设 `save_best`（v2 为 `bbox_mAP_50`）— [SARATR-X config](https://raw.githubusercontent.com/waterdisappear/SARATR-X/main/detection/configs/_hivit_/hivit_base_SARDet.py)、[v2 config](https://raw.githubusercontent.com/Pepu0309/SARATR-X-v2/main/results/detection/sardet/pixel_itpn_GFL_sardet100k.py)。v2 iTPN-L 表值与第 36 epoch 日志相符，未见挑 epoch 痕迹 — [log](https://raw.githubusercontent.com/Pepu0309/SARATR-X-v2/main/results/detection/sardet/large_20260421_033049.log.json)
- DenoDet V2 Table 5 从 stride 1/2/4/8/16 里选 mAP 最高的 8（56.7）作为最终结果 — [DenoDet V2](https://arxiv.org/html/2508.09392)；第三方评论称这是在测试集上选的 — [Pith](https://pith.science/paper/2508.09392)（搜索摘要，未核实）

**计算量 / 实时性**
- SARATR-X HiViT 66M、ViT 86M、ConvNeXt-V2 89M（Table IV）— [SARATR-X](https://arxiv.org/html/2405.09365v4)；v2、SARMAE、ViTP、BabelRS 均未见检测 FPS — [v2](https://arxiv.org/html/2607.23238)、[SARMAE](https://arxiv.org/html/2512.16635)、[ViTP](https://arxiv.org/html/2509.17562v3)、[BabelRS](https://arxiv.org/html/2603.01758)
- v2 预训练 1200 epoch、8×A800 — [v2](https://arxiv.org/html/2607.23238)；SARMAE batch 1024、A800 — [SARMAE](https://arxiv.org/html/2512.16635)
- MDIS-DETR FPS 从 187.93 降到 150.17，FLOPs 从 56.97 增到 66.39，而单空间域模块 FPS 141.80 低于完整模型，频域模块 FPS 198.87 高于基线（FLOPs 更高），FPS 测量稳定性存疑 — [Table 3](https://www.mdpi.com/2072-4292/18/13/2076)

**单检测器增益**
- MSFA 少见地覆盖 10 个检测器和 12 个主干（Table S8/S9）— [SARDet-100K](https://arxiv.org/html/2403.06534v3)；SARATR-X、v2、SARMAE、SAMBA 在 SARDet-100K 上都只报告一种头，未在 DETR 头上测 — [v2 §IV-A](https://arxiv.org/html/2607.23238)、[SARMAE §12.2](https://arxiv.org/html/2512.16635)
- MSFA 对 Deformable DETR 只 +1.3（AP50 仅 +0.2）— [Table S8](https://arxiv.org/html/2403.06534v3)

**数字不一致（同一方法跨表）**

| 方法 | 数值（mAP） | 来源 |
|---|---|---|
| GFL R50 | 49.8 / 55.01 / 34.1 | [MSFA](https://arxiv.org/html/2403.06534v3)、[DenoDet V2](https://arxiv.org/html/2508.09392)、[CGAQ-DETR 7 epoch](https://www.mdpi.com/2072-4292/17/18/3254) |
| Deformable DETR | 50.0 / 52.00 / 33.3 | 同上 |
| DETR | 31.8 / 45.73 / 7.1 | 同上 |
| DAB-DETR | 45.9 / 43.31 / 28.1 | 同上 |
| SARATR-X | 57.3 / 57.2 | [Table VII vs XII](https://arxiv.org/html/2405.09365v4) |
| DenoDet V2 | 56.71（AP50 85.86）/ 56.67（AP50 86.24）/ 56.4 | [原文](https://arxiv.org/html/2508.09392)、[MDIS Table 8](https://www.mdpi.com/2072-4292/18/13/2076)、[ViTP Table 2](https://arxiv.org/html/2509.17562v3) |
| MD-DETR AP50 | 88.5 / 87.96 | [摘要](https://ieeexplore.ieee.org/document/10741545)、[MDIS Table 8](https://www.mdpi.com/2072-4292/18/13/2076) |
| ViTP | 59.7 / 57.9 | [论文](https://arxiv.org/html/2509.17562v3)、[README](https://huggingface.co/GreatBird/ViTP/blob/main/README.md) |

- SARATR-X-v2 README 称预训练数据约 500K 切片，并描述为 FAIR-CSAR 大幅图 — [repo](https://github.com/Pepu0309/SARATR-X-v2)，论文写 17 源约 320K — [v2](https://arxiv.org/html/2607.23238)，两处表述不符，原因未说明。
- MDIS-DETR Table 8 中 GFL FLOPs 32.27（DenoDet V2 为 52.36），Conditional DETR AP75 51.67（DenoDet V2 为 44.40）— [MDIS](https://www.mdpi.com/2072-4292/18/13/2076)、[DenoDet V2](https://arxiv.org/html/2508.09392)
- SAR-NAS 把 DenoDet 引作 [19]，而 [19] 实为 DAB-DETR — [SAR-NAS](https://arxiv.org/html/2509.01279)

### Inferences
- MDIS-DETR Table 8 的 FCOS、GFL、ATSS 等 mAP 与 DenoDet V2 S7（12 epoch，512，IN 预训练）逐项相同，而它自身是 150 epoch 从零训练；其 CGAQ-DETR 行 38.62 又明显来自 7 epoch 协议。表面看是跨协议转抄，“超过 30 个 SOTA” 不是同协议比较。论文未声明这些行是否复跑。
- SAR-NAS Table 1 基线同样与 DenoDet V2 相同，而 SAR-NAS 的超网训练为 500 epoch（重训 epoch 未读到），69.27 与这些 12 epoch 基线的差距也不能全算到架构。
- 对论文写作，建议只引同文同协议内的差值，并在跨文比较时标明 epoch、输入、预训练、头。

### Gaps
- 多数基础模型论文的逐类 AP 未公开；SARATR-X-v2 日志有 `classwise=True`，但所读片段未见分类别值。
- FPS 的硬件、批量、是否含后处理在多数文章中未写明；未做统一测速。
