# SARDet-100K 没有总榜：先跑基线，再定路线

**SARDet-100K 上不存在可以合并的 SOTA 总榜，头名要按协议家族分开看。** 短日程路线中，split 明确的头名是 AAAI 2026 的 DenoDet V2（val，AP 56.71）。SAR 预训练路线在 test 上最高的是 SARATR-X-v2 iTPN-L（AP 68.7，发表状态未确认），顶会顶刊档最高的是 SARMAE ViT-L（AP 63.1，split 未写；CVPR 2026 据仓库标注）。YOLO 长日程的最高自报值是 SAR-NAS-S 69.27，但协议未报告，且 YOLO 结果分成两簇，64–69 不是普遍水平。实时 DETR 正文可证的最高值是 MDIS-DETR 58.82，更高的 WaveFormer RT-DETR 61.5 只读到摘要。**缺口还在，但最大的空间不在下一个频域模块，而在评测结构、跨来源泛化和 bridge/tank 的高 IoU 损失**，这些方向在检索范围内都没有量化工作。**基线建议走通用检测器路线：DEIM-D-FINE 做主基线（M 为主力，S 为闸门），GFL 和 DenoDet V2 只作自跑校准行，不在单篇 SAR 论文上做增量。** 这个建议有前提：你确认全集 baseline 从未在服务器上跑过，所以下文所有 DEIM-D-FINE 全集 AP 都是低置信度估计，跑出 baseline 是一切判断的先决条件。

**阅读约定。** 核验状态分五种：**复核**指普查阶段回到论文表格核对过；**正文已读**指笔记读过正文、配置或日志，普查未再核对；**转引**指只见于别的论文的表；**README** 指只见于仓库说明；**仅摘要**指 abstract only, not verified，不进入任何排名。AP 指 COCO AP@[.50:.95]，原文只写 mAP 且未说明阈值的标为“口径未写”。标“证据”的内容来自论文表格或本地文档，标“推断”的内容是本报告的判断，标“提议阈值”的数字不是文献标准，开跑前需要你确认。本报告只综合文献和仓库文档，没有训练、没有访问服务器。

## 建议：用 DEIM-D-FINE 做主基线，把 GFL 和 DenoDet V2 降为校准行

**证据。** 已读正文的 11 篇论文中，第三方方法都建立在通用检测器上（GFL、Faster/Cascade R-CNN、RT-DETR、Deformable DETR、YOLO、DEIM），在某个 SAR 方法上继续做的只有原作者续作（DenoDet→V2、SARATR-X→v2）。普查中最接近的反例是 [EO2SAR-Diff](https://doi.org/10.3390/rs18122037)，它把 DenoDet 作为检测宿主，但贡献是合成数据增广，不是模块增量；作者是否与 DenoDet 团队重叠，笔记没有记录。这个样本有限，TCSVT 和大部分 TGRS 正文没有读到，审稿意见也只能由已发表论文反推。

检索范围内**没有找到 D-FINE 或 DEIM 在 SARDet-100K 全集上的数字**。唯一相关记录是 [GrokSAR #41](https://github.com/GrokCV/GrokSAR/issues/41)：有人用 D-FINE-S 按 12 轮对标，没有给出数值。[SARES-DEIM](https://arxiv.org/html/2604.04127) 已在 HRSID 和 SAR-Ship 上用 DEIM-S 做基线，所以这个空位可能很快被占。

**推断。** DenoDet V2 路线的吸引力是真实的：每轮实验便宜约一个数量级，一个仓库内能用同一配置复现 GFL 和 V2，还有 AAAI 先例。但它的新颖性要建立在一个 +1.70 的模块上，其中 Harbor 一类贡献了 50.8%，配方上限也由原作者决定。[GrokSAR](https://github.com/GrokCV/GrokSAR) 的许可证声明也有冲突（README 写 CC BY-NC 4.0，页面显示 Apache-2.0）。SARATR-X 系的预训练需要 8 卡、800–1200 轮，v2 权重要邮件申请，无法独立复现。DEIM 路线的短板同样明确：消融慢、基线实力未知，第一档论文也还没有把实时 DETR 作为 SARDet-100K 的主要结果，空位和无先例是同一件事。“首个 D-FINE/DEIM 基准”本身也不足以成文，新意要来自下一节的缺口。

| 选项 | 单次成本（估计，误差可到 ±50%） | 代码与复现 | 全集 AP 的性质 | 论文空位 | 判断 |
|---|---|---|---|---|---|
| **DEIM-D-FINE-M** 640 | 43–67 h；单种子 6 组消融约 11–17 天 | Apache-2.0；项目基础设施已就绪；SARDet 的 M 配置尚未建立 | **低置信度估计 56–62，未实测** | 高，未见公开基准 | **主力** |
| DEIM-D-FINE-S 640 | 34–59 h | [配置和协议已定](../docs/sardet100k_full_baseline.md)，**从未运行** | **低置信度估计 55–60，未实测** | 高 | **先跑，作闸门和轻量行** |
| DEIM-D-FINE-L 640 | 35–54 h | 同上，官方仅 58 轮 | **低置信度估计 57–63，未实测** | 高 | 最终方法的规模对照 |
| GFL（MMDet） | 2–6 h | 环境锁定；55.01 依赖 DAdaptAdam 配方 | 文献 55.01（val，512，12e，复核） | 低 | **校准行** |
| 在 DenoDet V2 上增量 | 3–6 h | 许可证冲突；val 和 test 是两套配置 | 文献 56.71（val，复核） | 低 | **校准行，不做主线** |
| 在 SARATR-X / v2 上增量 | 8–15 h（把握低） | 预训练不可复现，权重需索取 | 文献 57.3 / 68.7（test，复核） | 中 | 不选；可探索其权重作初始化 |
| YOLO s | 12–25 h | 文献数字互相矛盾 | 文献 50–69 分两簇，**不能当期望值** | 低 | 不做主线；**应做一次校准** |
| RT-DETRv2 / v4 | 30–50 h（R18） | v4 需 teacher 权重 | 文献 54.24–58.2（split 未写） | 低 | 对照行 |

全部 AP 估计都依赖跨论文外推（RT-DETR 系 54–58，加 COCO 和其他 SAR 数据集上的规模增益），不是任何实验结果。**推断：**基线大概率进入 51–57 这一档，也可能超过，但很难单靠基线到 63 以上。除协议存疑的 YOLO 行外，全集 63 以上的结果都来自 SAR 预训练路线，同一方法从 B 换 L 就有约 +5。DEIM-D-FINE 跑出来后，最该先对照的是 RT-DETR 系 54–58 和 [Grad-YOLO 同表](https://doi.org/10.1109/JSTARS.2026.3692978)的 YOLO 56–57，然后才是 val 上的 DenoDet V2 56.71。本项目的 val.json 结果能直接对标的文献行只有四条 val 结果：DenoDet V2 56.71、DenoDet 55.88、GFL 55.01、Pu & Xu 54.9，其余都要写明差异。

## 没有总榜：四个家族、三种 split，各有各的头名

官方划分是 train 94,493 / val 10,492 / test 11,613（[Table 1](https://arxiv.org/html/2403.06534v3)）。test 已通过 [Issue #31](https://github.com/zcablii/SARDet_100K/issues/31) 公布，README 仍写只发布 train 和 val。普查中约一半行没有写 split，所以下面的名次只在同一组内成立。**跨组、跨表不能比较。** 普查共约 190 条候选，40 多篇有数值线索，30 余篇能落到表号。

### 已发表的顶会顶刊实际报告了什么

分档是笔记的操作定义，第一档为 CCF-A 类视觉与人工智能会议及 TPAMI、TIP，不是官方分级。

| 论文（2025–2026） | 发表 | 贡献类型 | 全集结果 | 状态 |
|---|---|---|---|---|
| [SARMAE](https://arxiv.org/abs/2512.16635) | CVPR 2026（据仓库，未到会议官网核对） | 噪声感知 MAE 预训练 | ViT-L 63.1，ViT-B 57.9，只有 AP | 复核 |
| [SARATR-X](https://arxiv.org/abs/2405.09365) | TIP 2025 | SAR MIM 预训练 | 57.3（另一表 57.2） | 复核 |
| [DenoDet V2](https://arxiv.org/html/2508.09392v1) | AAAI 2026 | 频域模块 | val 56.71 | 复核 |
| [MaRS](https://doi.org/10.1609/aaai.v40i14.38153) | AAAI 2026 | 多模态预训练 | 55.4 | 转引 |
| [RingMoE-KC](https://arxiv.org/abs/2504.03166)、[RingMo-Agent](https://arxiv.org/abs/2507.20776) | TPAMI | 多模态 | 只有 AP50（89.72、53.84） | 复核 |
| [DNOD](https://iclr.cc/virtual/2026/poster/10014070) | TMLR 2025，ICLR 2026 Journal Track | DETR 频域算子 | 只有 “+2.23 mAP” | 仅摘要 |

全集上 64–69 的数字没有一个来自这一档。[SM3Det](https://arxiv.org/abs/2412.20665) 的 65.20 是子集结果，不算。**推断：**投 CVPR、AAAI 档时，审稿人看到的上沿大概率是 56–63，不是 69。

### 家族一：短日程（12 轮），无 SAR 专用预训练

| 名次 | 方法 | 方法与协议 | AP / AP50 / AP75 / APs | 状态 |
|---|---|---|---:|---|
| val-1 | [DenoDet V2](https://arxiv.org/html/2508.09392v1) | GFL + FFTResNet-50，stem 和 4 个 stage 后做 DFT 幅度/相位交叉注意力；512，DAdaptAdam；32.60M | 56.71 / 85.86 / 61.55 / 51.45 | 复核；val 来自 README 与配置 |
| val-2 | [DenoDet](https://arxiv.org/abs/2406.02833) | GFL + R50，FPN 融合后做 DCT 域注意力；512；65.78M | 55.88 / 85.81 / 60.16 / 50.63 | 复核；val 由配置推断 |
| val-3 | [GFL](https://arxiv.org/abs/2406.02833) | R50，同上配方；32.27M | 55.01 / 85.16 / 58.87 / 49.44 | 复核 |
| val（自述） | [PGSNet](https://ace.ewapub.com/article/view/37773.pdf) | GFL + R50，物理模块；800；会议集 | 58.2 / 87.9 / 63.8 / 53.1 | 复核；val 口径与其表内基线不一致（见异常表），不计为 val 头名 |
| test-1 | [MSFA ConvNeXt-B](https://arxiv.org/html/2403.06534v3)（2024 参考） | Faster R-CNN，WST 输入，ImageNet→DOTA 预训练；800，取第 12 轮 | 56.4 / 88.2 / 61.5 / 51.1 | 复核；test 明写 |
| test-2 | [DenoDet V2 test 配置](https://github.com/GrokCV/GrokSAR) | 1024，stride 16，AdamW；0.21T | 56.39 / — / — / — | README |
| test-3 | [MSFA GFL](https://arxiv.org/html/2403.06534v3) | R50；ImageNet 版 49.8 | 53.7 / 84.2 / 57.8 / 47.8 | 复核 |
| 未写-1 | [UDCNet](https://doi.org/10.1109/JSTARS.2025.3585896) | 分割辅助双分支 + ConvNeXt-B；512；检测头未写明 | 59.3 / 89 / 65.4 / 54.1 | 复核（看图） |
| 未写-2 | [MSPaDet](https://doi.org/10.3390/rs18101513) | FPN + DTCWT 相位一致性；512；49.50M，101.9 FPS | 56.69 / 86.32 / 61.17 / 50.80（Table 5 写 0.5833） | 复核；见异常表 |

未入榜的有两篇：[EO2SAR-Diff](https://doi.org/10.3390/rs18122037) 60.00 / 88.10 / 63.70 / 57.20 靠合成增广，轮数未写；[TLTA](https://doi.org/10.3390/rs18193268) 56.9 / 89.1 的协议未读到。

### 家族二：SAR 或遥感预训练

| 名次 | 方法 | 方法与协议 | AP / AP50 / AP75 / APs | 状态 |
|---|---|---|---:|---|
| test-1 | [SARATR-X-v2 iTPN-L](https://arxiv.org/pdf/2607.23238) | GFL，36 轮，训练 384–640、测试 512；17 源约 320K 图预训练 1200 轮；seed 42 | 68.7 / 93.9 / 75.3 / 67.5（APs 来自日志） | 复核；配置 val 也指向 test.json，表值与第 36 轮日志相符。[arXiv 写投稿准备中](https://arxiv.org/pdf/2607.23238)，[OpenAlex 有 GRSM DOI](https://doi.org/10.1109/mgrs.2026.3738381)，**发表状态未确认** |
| test-2 | [SARATR-X-v2 iTPN-B](https://arxiv.org/pdf/2607.23238) | 同上 | 63.4 / 92.5 / 70.1 / — | 复核 |
| test-3 | [MMATR-X](https://arxiv.org/abs/2604.16952) | Faster R-CNN + HiViT-B，36 轮；OSPretrain-1M 预训练 800 轮 | 61.1 / — / — / — | 复核；test 明写；arXiv |
| test-4 | [SARATR-X](https://arxiv.org/abs/2405.09365) | Faster R-CNN + HiViT-B，36 轮（配置），480–800 多尺度 | 57.3 / 88.7 / 62.8（Table XII 57.2 / 88.5 / 62.6 / 51.2） | 复核；配置指向 test.json |
| val-1 | [SAR-LoRA-DINO](https://github.com/Jordan-Liao/SAR-LoRA-DINO) | RetinaNet + DINOv3 ConvNeXt-S，全微调 | 57.2 / — / — / — | README，正文未读 |
| val-2 | [Pu & Xu](https://arxiv.org/html/2501.11249v1) | ViTDet ViT-B，SARDet train 上 MAE 400 轮；512，12 轮；ImageNet 版 53.6 | 54.9 / 85.9 / 60.5 / 50.0 | 复核；val 明写 |
| 未写-1 | [SARMAE ViT-L](https://arxiv.org/abs/2512.16635) | Faster R-CNN，ViT 除末层冻结；SAR-1M 预训练；307M | 63.1 / — / — / — | 复核 |
| 未写-2 | [ViTP](https://arxiv.org/abs/2509.17562) | Cascade R-CNN + ViT-L，VLM 指令预训练；800 | 59.7（README 57.9） | 复核；arXiv |
| 未写-2 | [SAMBA](https://arxiv.org/abs/2606.31668) | 双向 Mamba + SG-MAE，检测头未写 | 59.7 / 89.1 / 68.0 / — | 复核；arXiv |
| 未写-4 | [SARVLMDet v4](https://arxiv.org/abs/2510.22665) | Faster R-CNN + ViT-B/16 | 57.4 / 89.4 / 63.4（[v3](https://arxiv.org/html/2510.22665v3) 为 60.2 / 92.6 / 66.9） | 复核；arXiv |
| 未读 | [SUMMIT](https://arxiv.org/abs/2606.31668)、[MaRS](https://arxiv.org/pdf/2607.23238) | ViT-B；SwinV2-B | 57.0 / 89.9 / 62.7；55.4 / — / 61.4 | 转引 |

### 家族三：YOLO 长日程，两簇不能合并

| 簇 | 方法 | 方法与协议 | AP / AP50 / AP75 | 状态 |
|---|---|---|---:|---|
| 高 | [SAR-NAS-S](https://arxiv.org/abs/2509.01279) | YOLOv10 通道宽度搜索；7.07M；输入、轮数、split 未写 | 69.27，口径未写 | 复核 |
| 高 | [YOLOv10-S（同文）](https://arxiv.org/html/2509.01279v1) | 7.22M，同上 | 68.84，口径未写 | 正文已读 |
| 高 | [SARLite](https://www.nature.com/articles/s41598-026-49143-5/tables/1) | YOLO11s + 轻量 HG-block + DySample；640，200 轮；7.8M | 67.7 / 92.8 / —；5 折 66.7±2.0 / 92.2±0.9 | 复核；split 未写，消融用 5 折 |
| 高 | [同表 YOLOv10-s、YOLO11-s、YOLOv8-s](https://www.nature.com/articles/s41598-026-49143-5/tables/1) | 8.0–9.8M | 65.1、64.0、63.7 | 复核 |
| 未归簇 | [GSFFE-Net](https://doi.org/10.1109/JSTARS.2026.3676453) | YOLO 系 + CSPDarknet；300 轮，无预训练；3.8M | 64.3 / 89.0 / 71.8 | 复核（看图）；test 由其 Table III 推断 |
| 低 | [Grad-YOLO](https://doi.org/10.1109/JSTARS.2026.3692978) | YOLOv8m 双流 + GCN；512，无预训练；27.4M | 57.6 / 90.6 / 65.9 | 复核（看图）；评测集未写 |
| 低 | [同表 YOLOv12m、YOLOv8m、YOLOv10m](https://doi.org/10.1109/JSTARS.2026.3692978) | 15.4–25.9M | 56.8、56.5、55.9 | 复核（看图） |
| 低 | [Edge YOLO](https://doi.org/10.3390/rs17132168) | YOLOv8n 改；640，300 轮；1.9M | 54.0 / 87.7 / — | 复核 Table 6 |
| 低 | [同表 YOLOv10n、v9t、v8n、v5n](https://doi.org/10.3390/rs17132168) | n/t 规模，不超过 3.2M | 53.1、52.4、51.8、49.6 | 复核 Table 6 |
| 低 | [LightKD-SAR](https://doi.org/10.1109/JSTARS.2026.3655160) | YOLOX-s 蒸馏；11.43M；基线 48.75 | 50.92，口径未写 | 复核（看图） |

**证据：**25.9M 的 YOLOv8m 在 512 下是 56.5，9.8M 的 YOLOv8-s 在 640 下是 63.7，规模更大反而低 7.2 点。约 8M 的 YOLOv10 在三篇论文里分别是 [54.1](https://doi.org/10.1109/JSTARS.2026.3676453)、[65.1](https://www.nature.com/articles/s41598-026-49143-5/tables/1)、[68.84](https://arxiv.org/html/2509.01279v1)，相差 14.7 点。**推断：**输入尺寸只能解释一部分，split、折划分或评测实现都可能参与，目前无法判定。在同一 split、同一 COCOeval 复跑之前，**64–69 不能当作 YOLO 的普遍水平**，也不能当作 DEIM 的对标线。

### 家族四：DETR 长日程

| 名次 | 方法 | 协议 | AP / AP50 / AP75 / APs | 状态 |
|---|---|---|---:|---|
| 1 | [MDIS-DETR](https://doi.org/10.3390/rs18132076) | RT-DETR + R18，加输入、编码器和跨尺度三类模块；640，150 轮，无预训练、无显式增强；22.42M，150.17 FPS | 58.82 / 89.31 / 63.68 / 53.43 | 复核；split 未写 |
| 2 | [RT-DETRv2-R18（Edge YOLO 表）](https://doi.org/10.3390/rs17132168) | 640，300 轮 | 58.2 / 90.2 / — / — | 复核；对比行 |
| 3 | [RT-DETR（Grad-YOLO 表）](https://doi.org/10.1109/JSTARS.2026.3692978) | 512，无预训练；36.0M | 57.6 / 90.2 / 65.9 / — | 复核；对比行 |
| 4 | [RT-DETR-R18（Edge YOLO 表）](https://doi.org/10.3390/rs17132168) | 640，300 轮 | 57.1 / 89.5 / — / — | 复核；对比行 |
| 5 | [RT-DETR 基线（MDIS 表）](https://doi.org/10.3390/rs18132076) | 640，150 轮，无预训练、无显式增强 | 54.24 / 85.89 / 58.41 / 48.74 | 复核 |
| 6 | [RT-DETR-R50（GSFFE 表）](https://doi.org/10.1109/JSTARS.2026.3676453) | 300 轮，无预训练 | 47.3 / — / — / — | 复核；是否自跑未核实 |

**只读到摘要或转引，不计入排名：**[WaveFormer RT-DETR](https://doi.org/10.1109/LGRS.2025.3646494) “mAP 61.5”，阈值未写；[SAR-DETR](https://doi.org/10.1109/TAES.2026.3710962) mAP 0.586；[CARL-RTDETR](https://doi.org/10.1109/TAP.2026.3656217) AP 56.1；[SA-RTDETR](https://doi.org/10.1109/LGRS.2025.3632153) AP 56.0、AP50 90.1；[MD-DETR](https://doi.org/10.1109/TGRS.2024.3489212) 摘要只有 AP50 88.5，[MDIS 表转引 55.38](https://doi.org/10.3390/rs18132076)。[CGAQ-DETR](https://doi.org/10.3390/rs17183254) 38.6 只训 7 轮，其 69.8 是 AP50，不参与比较。

### 按指标分开看：每个家族和 split 的头名

下表只比较所列范围内已读到的行，不含只读到摘要的结果，也不能跨行比较。

| 家族 / 口径 | AP | AP50 | AP75 | APs |
|---|---|---|---|---|
| 短日程 / val（DenoDet V2 Table S7） | DenoDet V2 56.71 | [AutoAssign 89.58](https://arxiv.org/html/2508.09392v1) | DenoDet V2 61.55 | DenoDet V2 51.45 |
| 短日程 / test（MSFA 表） | ConvNeXt-B 56.4 | ConvNeXt-B 88.2 | ConvNeXt-B 61.5 | ConvNeXt-B 51.1 |
| 短日程 / split 未写 | UDCNet 59.3 | TLTA 89.1（协议未读到），UDCNet 89 | UDCNet 65.4 | UDCNet 54.1 |
| 预训练 / test | v2 iTPN-L 68.7 | 93.9 | 75.3 | 67.5（日志） |
| 预训练 / val | SAR-LoRA-DINO 57.2（README）；Pu & Xu 54.9（复核） | Pu & Xu 85.9 | Pu & Xu 60.5 | Pu & Xu 50.0 |
| 预训练 / split 未写 | SARMAE ViT-L 63.1 | SARVLMDet v4 89.4（SUMMIT 89.9 为转引） | SAMBA 68.0 | 未报告 |
| YOLO 长日程 / 多数未写 | SAR-NAS-S 69.27 | SARLite 92.8 | GSFFE-Net 71.8 | 未报告 |
| DETR 长日程 / 多数未写 | MDIS-DETR 58.82 | RT-DETRv2-R18 与 RT-DETR 90.2 | RT-DETR（Grad-YOLO 表）65.9 | MDIS-DETR 53.43 |

[BabelRS](https://arxiv.org/abs/2603.01758) 的 63.30 / 91.70 已从全集榜移出。它和 SM3Det（60.64、62.98、65.20）同表，后者用 HRSID、MSAR、SADD、OGSOD、SIVED 五源子集（47,097 训练、4,481 测试），BabelRS 未单独说明，所以“五源子集”是由同表推断的。[SARATR-X-v2 Table X](https://arxiv.org/pdf/2607.23238) 把它和全集结果并列，引用时不要照搬。

### 同一协议的比较只有两种能找到

**证据：**[Grad-YOLO Table V](https://doi.org/10.1109/JSTARS.2026.3692978)（512，无预训练）中 RT-DETR 57.6、YOLOv12m 56.8、YOLOv8m 56.5、YOLOv10m 55.9；[Edge YOLO Table 6](https://doi.org/10.3390/rs17132168)（640）中 RT-DETRv2-R18 58.2、RT-DETR-R18 57.1、YOLOv10n 53.1。两张表都不支持“YOLO 明显强于 RT-DETR”，不过后一张的规模差距很大。反例是 [GSFFE Table V](https://doi.org/10.1109/JSTARS.2026.3676453) 的 RT-DETR-R50 47.3 低于 YOLOv8n 52.8，但该行是否自跑未核实。**GFL 与实时 DETR 的同协议比较没有找到**，也没有任何论文在同一 split、同一评测实现下同时跑 MMDet 系、YOLO 系和 DETR 系。

## 这些数字不能相加：协议差大于方法增益，对比列还有算术异常

下表是论文内部数字的差值。**推断：**模块增益明显小于预训练、骨干、日程和优化器组合造成的差异，所以小于 2 点的模块增益需要多种子才能宣称有效。模块、预训练和协议的增益来自不同协议，这里只比量级。

| 变化 | AP 差 | 来源 |
|---|---:|---|
| WST 输入（仅 ImageNet 预训练） | +0.2 | [MSFA Table 5](https://arxiv.org/html/2403.06534v3) |
| ImageNet → SAR MAE（ViTDet） | +1.3 | [Pu & Xu Table I](https://arxiv.org/html/2501.11249v1) |
| GFL → DenoDet V2 | +1.70 | [V2 Table S7](https://arxiv.org/html/2508.09392v1) |
| 12 → 36 轮（Faster R50） | +3.8 | [MSFA Table S9、S11](https://arxiv.org/html/2403.06534v3) |
| RT-DETR → MDIS-DETR（基线无预训练、无增强） | +4.58 | [MDIS Table 3](https://doi.org/10.3390/rs18132076) |
| ViT-B → ViT-L；iTPN-B → iTPN-L | +5.2；+5.3 | [SARMAE](https://arxiv.org/abs/2512.16635)、[v2](https://arxiv.org/pdf/2607.23238) |
| R50 → ConvNeXt-B（Faster） | +6.1 | [MSFA Table S9](https://arxiv.org/html/2403.06534v3) |
| 同一 GFL R50 跨协议（49.8 对 55.01） | 5.21 | [MSFA](https://arxiv.org/html/2403.06534v3)、[DenoDet](https://arxiv.org/abs/2406.02833) |
| 同一 Faster R50 跨协议（49.0 对 39.22） | 9.78 | 同上 |

DenoDet V2 的增益结构也值得注意。相对 GFL，**AP75 +2.68，AP50 只有 +0.70；Harbor +5.18，贡献 0.86 个 AP 点，占总增益 50.8%**，其余五类平均只有 +1.00（由 [Table S7](https://arxiv.org/html/2508.09392v1) 计算）。Harbor 在 SARDet-100K 中只来自 OGSOD。**推断：**论文的“去噪”叙事没有被这个类别分布很好地支持，增益的来源需要逐来源 AP 才能判断。

预训练路线同样无法把高分全部归因于预训练。[SARATR-X-v2](https://arxiv.org/pdf/2607.23238) 从 57.2 到 63.4 同时改了检测头、尺度、预训练目标和骨干。[SARATR-X 系预训练集](https://github.com/waterdisappear/SARATR-X)的来源覆盖 SARDet-100K 全部 10 个来源，作者声明移除了下游测试集，但没有公开排除清单。唯一给出具体数字的是 [MMATR-X](https://arxiv.org/abs/2604.16952)：预训练集与 test 有 160 张重叠，去除后 mAP 降 0.1；它不覆盖 train 与 test 之间的重复。这些都不是泄漏的证明，只说明隔离没有被独立验证。

下表只陈述**算术事实**。差异的成因笔记无法判定，本报告不作推断，也不对作者作任何判断。可操作的结论只有一条：**比较方法时只用同一论文内确实自跑的差值，对比行回到原文再用。**

| 论文与位置 | 算术观察 |
|---|---|
| [MSPaDet Table 4](https://doi.org/10.3390/rs18101513) 对 [DenoDet Table IV](https://arxiv.org/abs/2406.02833) | GFL、RepPoints、ATSS、PVT-T、TOOD、DDOD、YOLOF、YOLOX、ConvNeXt、ConvNeXtV2、DAB-DETR 共 11 行，12 个指标的差值都恰为 −0.30；FCOS 行差值为 0。MSAR 表抽查 FCOS 为 −1.30，SAR-Aircraft 表抽查 GFL 为 −5.50。自身结果同文有 56.69 和 0.5833 两个值 |
| [SARLite Table 1](https://www.nature.com/articles/s41598-026-49143-5/tables/1) 对 [MSFA Table S9](https://arxiv.org/html/2403.06534v3) | 四个骨干的 AP50 与 MSFA 行相同；AP 等于 MSFA AP 加括号增益：51.1+2.1=53.2，52.0+0.8=52.8，54.8+1.6=56.4，50.2+1.8=52.0 |
| [MDIS-DETR Table 8](https://doi.org/10.3390/rs18132076) 对 [V2 Table S7](https://arxiv.org/html/2508.09392v1) | 多数基线行相同；DenoDet V2 行为 56.67 / 86.24 / 61.02，原文 56.71 / 85.86 / 61.55；Conditional DETR 的 AP75 为 51.67，原文 44.40 |
| [PGSNet Table 1、3](https://ace.ewapub.com/article/view/37773.pdf) | 基线行等于 MSFA 的 test 值，正文声明基于 val；DenoDet V2 的 Tank、Bridge 为 58.3、57.1，原文 46.18、37.66 |
| [GSFFE Table V](https://doi.org/10.1109/JSTARS.2026.3676453) 对 [MSFA Table S8](https://arxiv.org/html/2403.06534v3) | Faster R-CNN 行 AP 为 41.1，AP50、AP75 为 82.2、52.9（MSFA 对应行 AP 49.0）；DETR 行 50.0 / 85.1 / 51.7 等于 MSFA 的 Deformable DETR 行 |
| [EO2SAR-Diff Table 3](https://doi.org/10.3390/rs18122037) 对 [DenoDet Table IV](https://arxiv.org/abs/2406.02833) | 标为 test 的基线 55.88 等于 DenoDet 的 val 行 |
| [Grad-YOLO Table V](https://doi.org/10.1109/JSTARS.2026.3692978) | DenoDet 行为 0.506，脚注称引自原文，原文为 55.88 |
| [UDCNet Table II、III](https://doi.org/10.1109/JSTARS.2025.3585896) | 对比行等于 MSFA 表的 test 值；正文起点写 51.5，Table III 为 51.2 |
| [SARVLMDet v3 对 v4](https://arxiv.org/abs/2510.22665) | 同一作者版本间 60.2 变为 57.4，差 2.8；[sota2 榜单](https://www.sota2.com/research/sota/object-detection-on-sardet-100k)仍列 60.2 |

## 缺口仍在：最大的空间是评测结构和高 IoU，不是下一个频域模块

**拥挤的方向。** 频域和去斑模块边际增益约 1–2 点，实时轻量化论文也多。本仓库已有多个模块实现后无收益的记录（SCAS、SBR、DPM、BCM、NWD、HWD，见 `docs/fdr_variance_step0.md:8`），[CLCR-Lite、W-FDR、PACFG 也已决定不再跟进](../docs/sar_dfine_paper_plan_2026-11-03.md)。这些记录多来自其他任务或子集，但足以说明不应把模块当主线。

**有证据的空白。** 第一，数据结构没有被审计。[SARDet-100K](https://arxiv.org/html/2403.06534v3) 由 10 个来源合并，四个来源占训练图 92.7%（自行计算），car 只来自 SIVED，harbor 只来自 OGSOD，切片有 200 像素重叠。[RSAR §4.1](https://arxiv.org/pdf/2501.04440) 明说 “duplicate and unannotated images remain”，但没有给出重复率。检索范围内也没有公开的跨划分近似重复率。**推断（粗略估算）：**如果 SAR-AIRcraft 上 82%、50% 的近似重复比例能迁移到切片层，仅这一个来源就约占 val 实例 16.7%、test 实例 12.5%；这个比例没有在 SARDet 切片上量过。

第二，高 IoU 损失的上限未知。AP50−AP 在各方法间稳定在 29–37 点，MDIS-DETR 距 100 分的差距中有 74% 落在这里，DenoDet V2 为 67%（自行计算）。bridge 最突出：[MSFA Table S13](https://arxiv.org/pdf/2403.06534) 的 AP50 .885，AP75 只有 .368。在 DenoDet V2 表中 mAP≥52 的 17 个方法上，bridge 均值 36.4、tank 42.8，都明显低于其他类（自行计算）。**但这只是一张表、一个模型的逐类 AP75，且没有任何 TIDE 式错误分解。** 这部分损失是真实定位误差还是标注模糊，目前无法区分。

第三，泛化协议缺失。[MSPaDet §4.2.4](https://doi.org/10.3390/rs18101513) 明确说自己的跨域实验不是严格的留一传感器评估，目标数据集没有可靠的逐图传感器标签；**推断：**它的目标子集本身含 SAR-AIRcraft 切片，所以那个 0.41 也不是纯粹的跨数据集结果。第四，协议本身也是缺口：多种子几乎没有；[SARATR-X、v2、SARMAE、SAMBA](https://arxiv.org/pdf/2607.23238) 都只报告一种检测头，没有测过 DETR 头；干净/加噪成对评测也未见。

**项目证据目前只到 10% 子集。** 常被引用的 **42.75 / 72.73 / 44.81 是混合值**：42.75 是旧协议日志 AP，72.73 和 44.81 是导出重算值，不应再使用。下面才是正确读法。**这些都是 DEIM-D-FINE-S 在 10% 子集（训练 9,449 张、筛选 val 4,096 张）上的结果，与任何全集结果都不可比。**

| 协议与来源 | AP | AP50 | AP75 | APs | 依据 |
|---|---:|---:|---:|---:|---|
| 固定协议，日志（**作基线引用**） | 43.06 | 72.83 | 45.10 | 35.74 | `docs/fdr_variance_step0.md:585` |
| 固定协议，导出重算（只用于诊断） | 43.18 | 72.93 | 45.26 | 35.78 | `docs/fdr_variance_step0.md:712` |
| 旧协议，日志（历史记录） | 42.75 | 72.63 | 44.59 | 35.78 | `docs/sardet_offset_check.md:351` |
| 旧协议，导出重算（只用于诊断） | 42.88 | 72.73 | 44.81 | 35.88 | `docs/sardet_offset_check.md:393` |

本地子集的 AP50−AP 为 29.77，与全量文献多数方法的 29–32 几乎相同，而 AP50 相差 13–16 点。**推断（方向性，模型、输入和 val 都不同）：**更多数据主要减少漏检和误检，对高 IoU 精度帮助有限。按固定协议导出值拆解，43.18 → 72.93（定位 +29.75，占差距 52%）→ 87.95（误检和排序 +15.02，26%）→ 100（无候选 +12.05，21%），顺序变了数字也会变（`docs/fdr_variance_step0.md:710-716`）。AP_GTBOX 的 +29.75 在机制上就是 AP50−AP，不是独立测量。旧协议下 tank、harbor、bridge 的 IoU≥0.5 匹配上限约为 76.9%、79.7%、81.8%（自行计算），只属于子集。系统偏移（oracle 约 +0.26）和逐边各向异性路线已可关闭；σ 与分数相关 −0.811，排序价值尚未测量（`docs/sigma_ranking_value.md` 只有设计）。

下表的排序基于“检索未见同类工作”，不是不存在的证明。证伪线都是**提议阈值，待你确认**，不是文献标准。所有审计只作诊断，保留官方划分和官方指标；重复检测只用相关性匹配（`tools/check_split_overlap.py`，24×24 相关，≥0.9，用于 SARDet 需小改），不使用哈希比对。

| # | 缺口 | 证据强度 | 成本（估计） | 判定 | 证伪线（提议阈值） |
|---|---|---|---|---|---|
| P0 | 全集 baseline | 前提，不是缺口 | S 34–59 h | **必须先有** | 见下一节 |
| E1 | 碎片 GT 与跨划分近似重复，按见过/没见过分组看 AP | 强 | 零训练 | **首选** | 碎片 GT <1% 且忽略后 AP 变化 <0.3；近似重复 <5% 且分组 AP 差 <3 |
| E2 | 高 IoU 的标注上限：RSAR 旋转框转水平框后与 SARDet 框比 IoU | 强 | CPU | **首选** | bridge IoU 中位数 <0.8 时下调定位方向；各类 >0.9 时定位方向成立。RSAR 是否沿用 SARDet 文件名未核实 |
| E3 | 逐来源 AP 与类别-来源混淆 | 中强 | 只需导出和评测 | **首选** | 同类别跨来源 AP 差 <5 |
| E4 | 留一来源泛化协议 | 中 | 子集每折一次 120 轮 | **主线候选** | 留出来源 AP 下降 <5 |
| E5 | tank/bridge/harbor 召回 | 中 | 依赖 P0 | 看 E1 | 全集 IoU≥0.5 匹配率都 >92% |
| E6 | 多来源合并后的漏标 | 弱到中 | 部分人工目视 | 看 E3 | 高分 FP 中疑似漏标 <10% |
| E7 | σ 排序增益 | 弱 | CPU，需写脚本 | 顺手关闭 | \|delta\| <0.5（`docs/sigma_ranking_value.md` 已预设） |
| E8 | 斑点干净/加噪评测 | 中 | 零训练起步 | 备选 | AP 下降 <1 |

**这份判断的边界。** 普查不是穷尽：TGRS、TAES、TAP、GRSL 的付费正文大多未读，CNKI、万方和 Google Scholar 未查，只在实验表里使用 SARDet-100K 的论文大概率还有遗漏。多数论文没有写 seed。所以“未见 DEIM 基准”“未见错误分解”都只是检索结果。决定这个建议的最大单一未知量是 DEIM-D-FINE 的全集实测 AP，而它现在一个数字都没有。

## 下一步，以及什么证据会推翻这个建议

先做的事很具体，每一步都有判定产物。小时数是估计，阈值是提议。

| 顺序 | 动作 | 产物与用途 |
|---|---|---|
| 1 | 确认本报告的提议阈值，并决定在哪台服务器跑。现成的 [`docs/sardet100k_full_baseline.md`](../docs/sardet100k_full_baseline.md) 含启动、恢复、日志和指标命令，但路径写的是新服务器 `/root/zxy`，而 `AGENTS.md` 默认旧服务器 | 决定路径与算力 |
| 2 | **运行 P0**：`configs/deim_dfine/deim_hgnetv2_s_sardet100k_baseline_120e_b32_paper_nozoom_nocrop.yml`，官方 train/val，120 轮，b32，seed 0，COCO 权重 `-t` | 实测 AP、s/iter，替换本文所有估计；随后重跑逐类 AP 和 `tools/localization_ceiling.py` |
| 3 | 与 P0 并行做 E1 的 JSON 统计和 E2，随后用现有子集检查点做 E3、E7 的预判，全集结果出来后重算 | 决定主线，不占训练 GPU |
| 4 | 在独立 MMDet 环境自跑 `gfl_r50_fpn_1x_SARDet_100k.py` 和 `DenoDetV2_1x_SARDet_100k.py`，预测导出 COCO JSON，与 DEIM 用**同一评估脚本**重算（各 2–6 h） | 回答 55.01、56.71 能否复现，建立唯一的同协议对照。写清过滤规则差异：GrokSAR `min_size=16`，本项目用完整 94,493 张 |
| 5 | 在官方 val 和同一 COCOeval 下跑一次 YOLO s（12–25 h） | 判定 64–69 是模型能力还是协议差异 |
| 6 | 补读全文：WaveFormer RT-DETR、SAR-DETR；核对 SARATR-X-v2 和 SARMAE 的发表状态；确认本地是否有 `test.json` | 确认最直接的竞争者及其协议 |
| 7 | S 过闸后新建 M 配置并跑全集（43–67 h），消融只在 M 上做，先子集筛选、再全集确认 | 主基线与消融。子集排名能否代表全集未被证明 |
| 8 | 按 E1–E3 定主线：审计结果显著，就写“评测结构与跨来源泛化”，配 E4；审计干净且 E2 显示标注足够好，转 bridge/tank 高 IoU 定位；E2 显示标注是上限，转检出和误检（E5、E6） | 论文主线 |
| 9 | 最终候选多种子（提议 3 个，待确认）；表格分三组：同协议自跑、官方 12 轮协议、文献报告值（标明 split，不参与排名）；若用 test，只评一次 | 可投稿的结果表 |

公开见过/没见过分组结果是否合适，由你决定；正式对比始终使用官方划分。

**反转条件，全部是提议阈值，待你确认：**

| 如果出现这个证据 | 那么 |
|---|---|
| S 全集 val AP 低于约 54（S 的闸门线，本报告提议，低于 MDIS 表 RT-DETR 的 54.24） | 先排查协议和数值问题，不跑 M |
| M 640 全集 val AP 明显低于约 54 | 基线不可用；排查后考虑转 GFL 路线 |
| 同一 val.json、同一评估下，自跑 GFL 比 DEIM-M 高 ≥2 AP，且 M 800 也追不上 | 基线故事不成立，转 GFL + 模块 |
| 单次 M 实测超过约 5 天，或可用 GPU 少于 2 张，或论文截止不足约 2 个月 | 成本不允许，转 MMDet 路线 |
| 目标期刊明确要求 12 轮同协议比较，且 DEIM 在 12 轮下明显落后 | 转 GFL 系 |
| 发现已发表或预印本的 D-FINE/DEIM on SARDet-100K，且高于本方案 | 空位消失，重写定位 |
| YOLO s 在官方 split、标准 COCOeval 下确实达到 64–69 | DEIM 档位偏低，重新评估主基线 |
| 能取得 SARATR-X-v2 权重，作为 DEIM 初始化后 AP 大幅提升 | 考虑“SAR 预训练 + 实时 DETR”主线（骨干适配成本高） |
| 自跑 GFL 复现不了 55.01（差 >1 AP） | **加强**本建议，并成为可写的发现 |
| E1、E3 同时触发证伪线 | 评测结构这条主线取消，转 E2 决定的方向 |
