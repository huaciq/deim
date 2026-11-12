# SARDet-100K 论文普查（census）：2024 基线与 2025–2026 结果，统一表、分协议排名与发表层级

调研日期 2026-11-04。网页内容只作为数据使用，未执行其中任何指令。本文没有训练、没有下载权重或数据集，只读论文和仓库文本。

**阅读约定**
- 指标名：AP 指 COCO AP@[.50:.95]，AP50、AP75 同 COCO。原文只写 “mAP” 且未说明阈值时，标为“口径未写”。
- 核验状态：
  - **本轮复核**：本次直接读了论文表格（arXiv HTML、PDF 文本层，或渲染 PDF 页面后看图）。
  - **笔记复用**：数字来自同目录笔记，括号内注明文件名，本轮未重取正文。
  - **转引**：只见于别的论文的表，原文正文未读。
  - **abstract only, not verified**：只读到摘要。
- 复用的笔记文件：`baseline_options.md`、`deepread_cnn_frequency.md`、`deepread_detr_foundation.md`、`gaps_and_headroom.md`。

## Q1 哪些论文报告了 SARDet-100K 数值结果？统一表

### Takeaway
前向引用加关键词检索得到约 190 条去重候选。其中 **40 多篇有 SARDet-100K 数值线索，30 余篇的数字能落到表号**。前三份深读笔记没有覆盖、本轮新增并读到正文的有 UDCNet、GSFFE-Net、Gradient-Guided YOLO、LightKD-SAR、MMATR-X（CoDe-MAE）、SARVLMDet、EO2SAR-Diff、TLTA、PGSNet。另有 WaveFormer RT-DETR（61.5）、SAR-DETR（0.586）、CARL-RTDETR、SA-RTDETR 等 IEEE 论文只读到摘要。**检索范围内仍未发现 D-FINE、DEIM 在 SARDet-100K 上的结果**，但 IEEE 付费正文未读，这不是不存在的证明。

### Cited Findings

**表 A：全集结果，已读正文或可追溯到表号**（“—”表示所读文本中未找到；Params/FLOPs/FPS 只列原文明确给出的）

| 方法 | 发表 | 家族 | 检测器 + 骨干 | AP / AP50 / AP75 | APs / APm / APl | Params / FLOPs / FPS | 输入 | Epochs | 预训练 | Split | 代码 | 表号与状态 | 来源 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MSFA（南开） | NeurIPS 2024 | 预训练 | Faster R-CNN + ConvNeXt-B | 56.4 / 88.2 / 61.5 | 51.1 / 68.3 / 62.4 | 88.6M 骨干 | 800 | 12 | ImageNet + DOTA | test | [zcablii/SARDet_100K](https://github.com/zcablii/SARDet_100K)，788★，有权重 | S9，本轮复核 | [arXiv](https://arxiv.org/html/2403.06534v3) |
| MSFA GFL | 同上 | 预训练 | GFL + R50 | 53.7 / 84.2 / 57.8（INP 49.8 / 80.9 / 53.3） | 47.8 / 66.2 / 59.5 | — | 800 | 12 | 同上 | test | 同上 | S8，本轮复核 | [arXiv](https://arxiv.org/html/2403.06534v3) |
| DenoDet | TAES（2024 在线） | 频域模块 | GFL + R50 | 55.88 / 85.81 / 60.16 | 50.63 / 68.47 / 60.96 | 65.78M / 52.69G | 512 | 12 | ImageNet | val（由配置推断） | [GrokSAR](https://github.com/GrokCV/GrokSAR)，115★，有权重 | IV，本轮复核；split 见 `deepread_cnn_frequency.md` | [arXiv](https://arxiv.org/abs/2406.02833) |
| GFL 基线（GrokSAR 协议） | 同上 | 通用 | GFL + R50 | 55.01 / 85.16 / 58.87 | 49.44 / 67.29 / 60.45 | 32.27M / 52.36G | 512 | 12 | ImageNet | val | 同上 | IV，本轮复核 | [arXiv](https://arxiv.org/abs/2406.02833) |
| DenoDet V2 | AAAI 2026 | 频域模块 | GFL + FFTResNet-50 | 56.71 / 85.86 / 61.55 | 51.45 / 68.75 / 61.34 | 32.60M / 52.47G | 512 | 12 | ImageNet | val | GrokSAR，有权重 | S7，本轮复核 | [arXiv](https://arxiv.org/html/2508.09392v1) |
| DenoDet V2（test 配置） | 同上，仓库 | 频域模块 | 同上，AdamW、stride 16 | 56.39（只有 AP） | — | 0.21T | 1024 | 12 | ImageNet | test | GrokSAR | README，笔记复用（`deepread_cnn_frequency.md`） | [GrokSAR](https://github.com/GrokCV/GrokSAR) |
| Pu & Xu MAE | IGARSS 2025 | 预训练 | ViTDet + ViT-B | 54.9 / 85.9 / 60.5（ImageNet 53.6；从零 47.2） | 50.0 / 65.3 / 58.7 | — | 512 | 12 | SARDet train 上 MAE 400e | **val（原文明写）** | — | I，本轮复核 | [arXiv](https://arxiv.org/html/2501.11249v1) |
| SARATR-X | TIP 2025 | 基础模型 | Faster R-CNN + HiViT-B | VII：57.3 / 88.7 / 62.8；XII：57.2 / 88.5 / 62.6 | 51.2 / 69.6 / 64.7 | 66M 骨干 | 训练 480–800 多尺度 | 36（配置） | ImageNet → SAR MIM | test（配置 val 也指向 test.json） | [SARATR-X](https://github.com/waterdisappear/SARATR-X)，277★ | VII、XII，本轮复核；配置见 `deepread_detr_foundation.md` | [arXiv](https://arxiv.org/abs/2405.09365) |
| SARATR-X-v2 | arXiv 头写投稿准备中；[OpenAlex 有 IEEE GRSM DOI](https://doi.org/10.1109/mgrs.2026.3738381) | 基础模型 | GFL + iTPN-L / iTPN-B | L：68.7 / 93.9 / 75.3；B：63.4 / 92.5 / 70.1 | L：67.5 / 77.2 / 71.5（日志） | — | 训练 384–640，测试 512 | 36（配置） | 17 源约 320K，1200e | test.json | [SARATR-X-v2](https://github.com/waterdisappear/SARATR-X-v2)，15★，权重需邮件 | X，本轮复核；配置、日志见 `deepread_detr_foundation.md` | [arXiv PDF](https://arxiv.org/pdf/2607.23238) |
| SARMAE | CVPR 2026（据 `baseline_options.md`） | 基础模型 | Faster R-CNN + ViT-L / ViT-B，ViT 除末层冻结 | L：63.1；B：57.9（只有 AP） | — | ViT-L 307M、ViT-B 86M | — | — | SAR-1M，含 SARDet train | 沿用 SARATR-X 协议，未明写 | [SARMAE](https://github.com/MiliLab/SARMAE)，65★，未见微调权重 | 3，本轮复核 | [arXiv](https://arxiv.org/abs/2512.16635) |
| MMATR-X（CoDe-MAE） | arXiv 2026-04 | 基础模型 | Faster R-CNN + HiViT-B | 61.1（只有 AP）；去除 160 张重叠后降 0.1 | — | 79M 骨干 | — | 36（12e 消融 53.5，Table XI，预训练轮数不同） | OSPretrain-1M，800e | **test（原文明写）** | [MMATR-X](https://github.com/scenarri/MMATR-X)，8★ | III、XI，本轮复核 | [arXiv](https://arxiv.org/abs/2604.16952) |
| ViTP（南开） | arXiv 2025 | 基础模型 | Cascade R-CNN + ViT-L | 59.7（README 57.9） | — | — | 800 | 默认值未写；另有一项延长到 36e 的研究 | VLM 指令预训练 | 未写 | ViTP，179★，有权重（`baseline_options.md`） | 2，本轮复核 | [arXiv](https://arxiv.org/abs/2509.17562) |
| SAMBA | arXiv 2026-06 | 基础模型 | 检测头未写 + 双向 Mamba | 59.7 / 89.1 / 68.0 | — | 约 27M、约 8G（骨干，笔记） | — | — | SG-MAE | 未写 | [SAMBA](https://github.com/mynswkk/SAMBA)，2★，论文称含权重 | IV，本轮复核 | [arXiv](https://arxiv.org/abs/2606.31668) |
| SARVLMDet | arXiv 2025-10 | 基础模型 | Faster R-CNN + ViT-B/16 | **v4：57.4 / 89.4 / 63.4；v3：60.2 / 92.6 / 66.9** | — | — | — | — | SARVLM-1M | 脚注称预训练不含检测测试集，未明写 | [SARVLM](https://github.com/KlayMa527/SARVLM)，19★ | VI，v3、v4 均本轮复核 | [v4](https://arxiv.org/abs/2510.22665)、[v3](https://arxiv.org/html/2510.22665v3) |
| SUMMIT | JAG 2025 | 基础模型 | ViT-B | 57.0 / 89.9 / 62.7 | — | — | 448（笔记） | — | MuSID | 未读 | [SUMMIT](https://github.com/Yunsans/SUMMIT)，14★ | 转引（SAMBA IV、v2 X） | [SAMBA](https://arxiv.org/abs/2606.31668) |
| MaRS | AAAI 2026 | 基础模型 | SwinV2-B | 55.4 / — / 61.4 | — | — | — | — | 多模态 | 未读 | — | 转引（v2 X） | [v2](https://arxiv.org/pdf/2607.23238) |
| SAR-LoRA-DINO | IGARSS 2026 | 预训练 | RetinaNet + FPN + DINOv3 ConvNeXt-S | 全微调 57.2；LoRA + stage3 56.9；纯 LoRA 52.3±2.8（3 seed） | — | 可训练 11.6–58.9M | — | — | DINOv3 | val | [repo](https://github.com/Jordan-Liao/SAR-LoRA-DINO)，1★ | **README，论文正文未读** | [README](https://github.com/Jordan-Liao/SAR-LoRA-DINO) |
| UDCNet（国防科大） | JSTARS 2025 | 分割辅助 | 双分支 + ConvNeXt-B，检测头未写明 | 59.3 / 89 / 65.4（Origin 55.1 / 87.8 / 59.5） | 54.1 / 70.8 / 64.7 | — | 512 | 12 | 未写 | 未写 | — | I、II、III，本轮复核（看图） | [DOI](https://doi.org/10.1109/JSTARS.2025.3585896) |
| EO2SAR-Diff | Remote Sensing 2026 | 数据增广 | GFL + R50 + DenoDet | 60.00 / 88.10 / 63.70（基线 55.88） | 57.20 / 71.50 / 65.80 | — | 训练图 512 | — | — | 原文写 test；基线行等于 DenoDet 的 val 行 | — | 3，本轮复核 | [MDPI](https://doi.org/10.3390/rs18122037) |
| TLTA | Remote Sensing 2026 | 拓扑模块 | 基线未在所读片段说明 | 56.9 / 89.1（基线 52.6 / 83.1） | — | — | — | — | — | 未写 | — | 16，本轮复核 | [MDPI](https://doi.org/10.3390/rs18193268) |
| PGSNet（中国海大） | ACE 会议集 2026 | 物理模块 | GFL + R50 | 58.2 / 87.9 / 63.8 | 53.1 / 70.4 / 65.2 | 36.9 FPS | 800 | 12 | — | 原文写 val | — | 1、2，本轮复核 | [PDF](https://ace.ewapub.com/article/view/37773.pdf) |
| MSPaDet | Remote Sensing 2026 | 频域模块 | FPN 系，骨干见原文 | **Table 4：56.69 / 86.32 / 61.17；Table 5：0.5833** | 50.80 / 68.30 / 62.50 | 49.50M / 64.60G / 101.9 FPS | 512 | 12 | ImageNet | 未写 | 按需提供 | 4、5，本轮复核；异常见 `deepread_cnn_frequency.md` | [MDPI](https://doi.org/10.3390/rs18101513) |
| MDIS-DETR（哈工程） | Remote Sensing 2026 | RT-DETR 系 | RT-DETR + R18 | 58.82 / 89.31 / 63.68（基线 54.24 / 85.89 / 58.41） | 53.43 / 71.37 / 67.08 | 22.42M / 66.39G / 150.17 FPS | 640 | 150 | 无，且无显式增强 | 未写 | — | 3、8，本轮复核 | [MDPI](https://doi.org/10.3390/rs18132076) |
| CGAQ-DETR | Remote Sensing 2025 | Deformable DETR 系 | Deformable DETR + R50 | 38.6 / 69.8 / 39.7 | 31.8 / 49.1 / 40.6 | — | 640 | **7** | — | 未写 | — | 4，本轮复核 | [MDPI](https://doi.org/10.3390/rs17183254) |
| SAR-NAS（北大） | PRCV 2025（据 [Semantic Scholar 元数据](https://api.semanticscholar.org/graph/v1/paper/arXiv:2403.06534/citations)） | YOLO | YOLOv10 搜索版 S / N | S：69.27；N：60.18（只有 AP，口径未写；同文 YOLOv10-S 68.84、N 59.47） | — | S：7.07M / 20.17G；N：1.97M / 5.96G | 未写 | 未写（超网 500） | 未写 | 未写 | 未公开 | 1，本轮复核；2 见 `deepread_cnn_frequency.md` | [arXiv](https://arxiv.org/abs/2509.01279) |
| SARLite | Sci. Rep. 2026 | YOLO | YOLO11s 改 | Table 1：67.7 / 92.8；Table 3（5 折）：66.7±2.0 / 92.2±0.9 | — | 7.8M / 20.3G | 640 | 200 | 未声明 | 未写；消融用 5 折 | — | 1 本轮复核；3 见 `deepread_cnn_frequency.md` | [Nature](https://www.nature.com/articles/s41598-026-49143-5/tables/1) |
| GSFFE-Net | JSTARS 2026 | YOLO | YOLO 系 + CSPDarknet | 64.3 / 89.0 / 71.8（基线 54.6 由 Table III 算出） | — | 3.8M，7.60MB | — | 300 | **无** | Table III 的基线实验明确用 test，推断 64.3 为 test | — | V，本轮复核（看图） | [DOI](https://doi.org/10.1109/JSTARS.2026.3676453) |
| Gradient-Guided YOLO | JSTARS 2026 | YOLO | YOLOv8m 双流 + GCN | 57.6 / 90.6 / 65.9 | — | 27.4M / 65.5G | 512 | — | **无** | 称用预划分集，未写评测集 | — | V，本轮复核（看图） | [DOI](https://doi.org/10.1109/JSTARS.2026.3692978) |
| Edge YOLO | Remote Sensing 2025 | YOLO | YOLOv8n 改 | 54.0 / 87.7 | — | 1.9M / 4.4G；TX2 34.2 FPS | 640（另有 320：42.8 / 67.3） | 300（`baseline_options.md`） | — | 未写 | — | 6，本轮复核 | [MDPI](https://doi.org/10.3390/rs17132168) |
| LightKD-SAR | JSTARS 2026 | 蒸馏 | YOLOX-s（教师 YOLOX-l 51.33） | 50.92（学生基线 48.75）（只有 AP，口径未写） | — | 11.43M / 15.7G | — | — | — | 未写 | — | III、IV，本轮复核（看图） | [DOI](https://doi.org/10.1109/JSTARS.2026.3655160) |

**表 B：只读到摘要（abstract only, not verified），不进入“已核实”排名**

| 方法 | 发表 | 摘要给出的数字 | 来源 |
|---|---|---|---|
| WaveFormer + WS-GD RT-DETR（中科院） | GRSL 2026 | “mAP 61.5%”，阈值未写 | [DOI](https://doi.org/10.1109/LGRS.2025.3646494) |
| SAR-DETR（华东师大） | TAES 2026 | mAP 0.586，22.94M | [DOI](https://doi.org/10.1109/TAES.2026.3710962) |
| CARL-RTDETR（安徽大学） | TAP 2026；同文会议版 APMC 2025 | AP 56.1、AP75 61.0，相对 RT-DETR50 +3.0、+4.6，参数少 47.2% | [TAP](https://doi.org/10.1109/TAP.2026.3656217)、[APMC](https://doi.org/10.1109/APMC65046.2025.11377600) |
| SA-RTDETR（安徽大学） | GRSL 2026；会议版 ACES-China 2025 | AP50 90.1、AP 56.0、recall 84.7，相对基线 +2.7、+2.6、+2.2 | [GRSL](https://doi.org/10.1109/LGRS.2025.3632153)、[ACES](https://doi.org/10.23919/ACES-China66523.2025.11333210) |
| CA-HLD（同济） | SPIE RSMG 2026 | AP 52.0 → 55.1 | [DOI](https://doi.org/10.1117/12.3130475) |
| MD-DETR | TGRS 2024（基线参考） | AP50 88.5；[MDIS Table 8 转引 55.38 / 87.96 / 59.08](https://doi.org/10.3390/rs18132076) | [DOI](https://doi.org/10.1109/TGRS.2024.3489212) |
| SA-DETR | JSTARS 2024（基线参考） | AP50 88.4 | [DOI](https://doi.org/10.1109/JSTARS.2024.3481368) |
| DNOD | TMLR 2025，ICLR 2026 Journal Track | +2.23 mAP；56 epochs、R50（配置，`deepread_cnn_frequency.md`） | [ICLR](https://iclr.cc/virtual/2026/poster/10014070) |
| YOLO + Lee 滤波（Binus） | ICoCICs 2025 | YOLOv11 滤波后 AP50 约 79（摘要片段截断） | [DOI](https://doi.org/10.1109/ICoCICs68032.2025.11383931) |
| SP-DETR（西电） | TGRS 2026 | 摘要称最佳，无数字 | [DOI](https://doi.org/10.1109/TGRS.2026.3729565) |
| DSA-YOLO（北理工） | IET 会议 2026 | 无数字 | [DOI](https://doi.org/10.1049/icp.2026.0567) |
| Frequency-Gated Decoupled-Synergy Net（南邮） | Remote Sensing Letters 2026 | 无数字 | [DOI](https://doi.org/10.1080/2150704x.2026.2727103) |
| Dynamic Frequency Convolution | 中文期刊 2026 | 无数字 | [DOI](https://doi.org/10.16356/j.2097-6771.2026.01.016) |
| Joint Spatial-Channel + Frequency Selection | 中文期刊 2026 | OpenAlex 命中，摘要未取到 | [OpenAlex 检索](https://api.openalex.org/works?search=SARDet-100K) |
| MSFA + ATSS | IJCA 2025 | 无数字 | [DOI](https://doi.org/10.5120/ijca2025925462) |
| CAVE-RTDETR | GRSL 2026 | OpenAlex 无摘要 | [DOI](https://doi.org/10.1109/LGRS.2026.3688978) |
| YOLOv8 on SARDet-100K | CISS 2025 | OpenAlex 无摘要 | [DOI](https://doi.org/10.1109/ciss67974.2025.11483030) |
| SARDet-CL、SARDetCL | TGRS 2025、ICIGP 2026 | 摘要无 SARDet-100K 数字；前者面向飞机检测 | [TGRS](https://doi.org/10.1109/TGRS.2025.3568041)、[ICIGP](https://doi.org/10.1109/ICIGP68997.2026.11619799) |
| Denoising-guided SAR foundation model、EvidenceSAR、PGCI-SAR、CSSCNet | ISPRS 2027、SSRN、KBS 2026、CCIS 2025 | OpenAlex 无摘要，是否报告 SARDet-100K 未知 | [ISPRS](https://doi.org/10.1016/j.isprsjprs.2026.10.019)、[SSRN](https://doi.org/10.2139/ssrn.7669955)、[KBS](https://doi.org/10.1016/j.knosys.2026.116846)、[CCIS](https://doi.org/10.1007/978-981-96-9946-9_21) |
| IRT-DETR、DIPKD | J. Supercomputing 2026、AAAI | 只有搜索摘要，见 `deepread_detr_foundation.md`、`deepread_cnn_frequency.md` | [ResearchGate](https://www.researchgate.net/publication/399890033_IRT-DETR_a_lightweight_transformer_with_ConvNextV2_and_Cascaded_Group_Attention_for_real-time_military_aircraft_detection_in_aerial_imagery) |

**表 C：子集或非标准任务，不能与全集比较**

| 方法 | 发表 | 数字 | 为何不可比 | 来源 |
|---|---|---|---|---|
| SM3Det | AAAI（据 Semantic Scholar 元数据） | ConvNeXt-T 60.64 / 89.94 / 65.06；ConvNeXt-S 62.98 / 91.49 / 68.77；ConvNeXt-B 65.20 / 92.41 / 70.02；同骨干单模型 GFL 57.31、60.62、62.27 | **只用 HRSID、MSAR、SADD、OGSOD、SIVED 五源子集**，47,097 训练、4,481 测试，与 DOTA、DroneVehicle 联合训练。Table 1、8 本轮复核 | [arXiv](https://arxiv.org/abs/2412.20665) |
| BabelRS | arXiv 2026 | 63.30 / 91.70（同表 SM3Det 60.64、Simple Joint 53.46） | SOI-Det 基准，同上。Table 2、3 本轮复核 | [arXiv](https://arxiv.org/abs/2603.01758) |
| Uncertainty-Aware MoE、Swin-MoE | TGRS 2026、SPIE ICDIP 2026 | 后者 Overall 51.82（三模态总分） | SOI-Det。abstract only | [TGRS](https://doi.org/10.1109/TGRS.2026.3707549)、[SPIE](https://doi.org/10.1117/12.3119573) |
| RingMoE-KC、RingMo-Agent | TPAMI | AP50 89.72（Table A15）；AP50 53.84（Table 15） | 只有 AP50；多模态 VLM 任务。本轮复核 | [RingMoE](https://arxiv.org/abs/2504.03166)、[RingMo-Agent](https://arxiv.org/abs/2507.20776) |
| Enhanced YOLO26 | SPIE RSMG 2026 | AP50 76.6 | 1 万张子集，舰船。abstract only | [DOI](https://doi.org/10.1117/12.3130570) |
| S2M-Net | Remote Sensing 2025 | AP50 0.883 | 1000 张纯 ship 子集（`deepread_cnn_frequency.md`） | [MDPI](https://www.mdpi.com/2072-4292/17/19/3347) |
| DGD | CAC 2024 | 摘要写 “SARDet-100K (part)” | 子集。abstract only | [DOI](https://doi.org/10.1109/CAC63892.2024.10865046) |
| RSAR、WE-KAN 等 OBB 工作 | CVPR 2025 等 | RSAR 表见 `deepread_detr_foundation.md` | 旋转框，另一任务 | [RSAR](https://arxiv.org/abs/2501.04440) |
| Geo-Topology-Guided Diffusion | Remote Sensing 2026 | — | 用 SAM-2 从 SARDet-100K 水平框自建 OBB，本轮只读到数据集段 | [MDPI](https://doi.org/10.3390/rs18183128) |

**只在引言或参考文献提及，无 SARDet-100K 结果（已查正文）**：[DA-GDNet](https://doi.org/10.3390/rs18162839)、[FSDNet](https://doi.org/10.3390/rs18121888)（结论称“仍需在 SARDet-100K 上验证”）、[Context-Guided Aircraft](https://doi.org/10.3390/rs18162756)、[Context-Driven Colored SAR](https://doi.org/10.3390/s26175427)、[LSDA-YOLO](https://doi.org/10.3390/sym18010023)、[FSMD-Net](https://doi.org/10.3390/rs18081254)、[Context-Guided Ship](https://doi.org/10.3390/s25164938)、[FUSAR-KLIP](https://arxiv.org/abs/2509.23927)、[SAR-Net](https://arxiv.org/abs/2312.16943)、[M4-SAR](https://arxiv.org/abs/2505.10931)。其中 [MaDiNet](https://arxiv.org/abs/2411.07500) 的判断来自 `deepread_detr_foundation.md`。

**摘要明确用了别的数据集，已排除**：[ShipGen](https://doi.org/10.1109/JSTARS.2026.3693312)、[SAR-RAFNet](https://doi.org/10.1109/jstars.2026.3729849)、[LDS-YOLO](https://doi.org/10.1088/1361-6501/aeabc6)、[ASC-U2Det](https://doi.org/10.1109/TGRS.2025.3550867)、[MSOD](https://doi.org/10.1109/TGRS.2025.3605720)。

### Inferences
- 引用 SARDet-100K 的论文大多不报告其结果。Semantic Scholar 153 条里，多数是 VLM、ATR 分类和数据集论文。
- 表 A 共约 30 行，真正同时给全 AP、AP50、AP75 的约 15 行。SARMAE、MMATR-X、ViTP、SAR-NAS 只给 AP，无法做 AP75 比较。

### Gaps
- TGRS、TAES、TAP、GRSL 的付费正文未读，表 B 的 61.5、0.586、56.1、56.0 都没有表号。
- 机构只在读到时填写，多数行未摘录。OpenAlex 只取了第一作者机构，可能不准。
- 多数论文未写 seed，也未读其权重发布情况，代码列只反映已查到的部分。

## Q2 同一指标、同一协议下，真正的榜首是什么？

### Takeaway
**不存在一个总榜。** 四个家族的头名分别是：短日程通用/模块路线 **UDCNet 59.3**（split 未写）或 val 明确的 **DenoDet V2 56.71**；SAR 预训练路线 **SARATR-X-v2 iTPN-L 68.7**；YOLO 长日程 **SAR-NAS-S 69.27**（协议未报告）；DETR 长日程全文可证的是 **MDIS-DETR 58.82**，摘要里有更高的 61.5。69.27 和 68.7 都不能直接当作可达上限。

### Cited Findings

**排名 1：AP，短日程（12–36e），无 SAR 专用预训练**（✓ 为本轮对原表复核）

| 排名 | 方法 | AP | Split | 输入 / Epochs | 备注 |
|---|---|---|---|---:|---|
| 1 | ✓ [UDCNet](https://doi.org/10.1109/JSTARS.2025.3585896) | 59.3 | 未写 | 512 / 12 | Table II 的对比行等于 MSFA 表的 test 值；正文说起点 51.5，Table III 为 51.2 |
| 2 | ✓ [PGSNet](https://ace.ewapub.com/article/view/37773.pdf) | 58.2 | val（自述） | 800 / 12 | 低层级；数据可信度见 Q3 |
| 3 | ✓ [DenoDet V2](https://arxiv.org/html/2508.09392v1) | 56.71 | val | 512 / 12 | |
| 4 | ✓ [MSPaDet](https://doi.org/10.3390/rs18101513) | 56.69 | 未写 | 512 / 12 | 对比列异常，同文 58.33 |
| 5 | ✓ [MSFA ConvNeXt-B](https://arxiv.org/html/2403.06534v3) | 56.4 | test | 800 / 12 | 含 DOTA 检测预训练 |
| 6 | [DenoDet V2 test 配置](https://github.com/GrokCV/GrokSAR) | 56.39 | test | 1024 / 12 | README 值 |
| 7 | ✓ [DenoDet](https://arxiv.org/abs/2406.02833) | 55.88 | val | 512 / 12 | |
| 8 | ✓ [GFL](https://arxiv.org/abs/2406.02833) | 55.01 | val | 512 / 12 | |

未入榜：[EO2SAR-Diff 60.00](https://doi.org/10.3390/rs18122037) 靠合成数据增广，轮数未写；[TLTA 56.9](https://doi.org/10.3390/rs18193268) 协议未读到。

**排名 2：AP，SAR 或遥感预训练路线**

| 排名 | 方法 | AP | 头 / Epochs | Split |
|---|---|---:|---|---|
| 1 | ✓ [SARATR-X-v2 iTPN-L](https://arxiv.org/pdf/2607.23238) | 68.7 | GFL / 36 | test.json |
| 2 | ✓ [SARATR-X-v2 iTPN-B](https://arxiv.org/pdf/2607.23238) | 63.4 | GFL / 36 | test.json |
| 3 | ✓ [SARMAE ViT-L](https://arxiv.org/abs/2512.16635) | 63.1 | Faster / 未写 | 未写 |
| 4 | ✓ [MMATR-X HiViT-B](https://arxiv.org/abs/2604.16952) | 61.1 | Faster / 36 | test |
| 5 | ✓ [ViTP ViT-L](https://arxiv.org/abs/2509.17562) | 59.7 | Cascade / 未写 | 未写 |
| 5 | ✓ [SAMBA](https://arxiv.org/abs/2606.31668) | 59.7 | 未写 | 未写 |
| 7 | ✓ [SARMAE ViT-B](https://arxiv.org/abs/2512.16635) | 57.9 | Faster / 未写 | 未写 |
| 8 | ✓ [SARVLMDet v4](https://arxiv.org/abs/2510.22665) | 57.4 | Faster / 未写 | 未写 |
| 9 | ✓ [SARATR-X](https://arxiv.org/abs/2405.09365) | 57.3（另一表 57.2） | Faster / 36 | test |
| 10 | [SUMMIT](https://arxiv.org/abs/2606.31668)（转引） | 57.0 | 未读 | 未读 |
| 11 | [MaRS](https://arxiv.org/pdf/2607.23238)（转引） | 55.4 | 未读 | 未读 |
| 12 | ✓ [Pu & Xu](https://arxiv.org/html/2501.11249v1) | 54.9 | ViTDet / 12 | val |

**排名 3：AP，YOLO 长日程（口径多数未写，Ultralytics 评测未确认）**

| 排名 | 方法 | AP | 输入 / Epochs | 规模 |
|---|---|---:|---|---|
| 1 | ✓ [SAR-NAS-S](https://arxiv.org/abs/2509.01279) | 69.27 | 未写 | 7.07M |
| 2 | [YOLOv10-S（同文）](https://arxiv.org/html/2509.01279v1)（`deepread_cnn_frequency.md`） | 68.84 | 未写 | 7.22M |
| 3 | ✓ [SARLite](https://www.nature.com/articles/s41598-026-49143-5/tables/1) | 67.7 | 640 / 200 | 7.8M |
| 4 | ✓ [YOLOv10-s（SARLite 表）](https://www.nature.com/articles/s41598-026-49143-5/tables/1) | 65.1 | 640 / 200 | 8.0M |
| 5 | ✓ [GSFFE-Net](https://doi.org/10.1109/JSTARS.2026.3676453) | 64.3 | — / 300 | 3.8M |
| 6 | ✓ YOLO11-s、YOLOv8-s（SARLite 表） | 64.0、63.7 | 640 / 200 | 9.4M、9.8M |
| 7 | ✓ [Grad-YOLO](https://doi.org/10.1109/JSTARS.2026.3692978)、YOLOv12m、YOLOv8m、YOLOv10m（同表） | 57.6、56.8、56.5、55.9 | 512 / 未写 | 27.4M、20.2M、25.9M、15.4M |
| 8 | ✓ [Edge YOLO](https://doi.org/10.3390/rs17132168)、YOLOv10n、YOLOv9t、YOLOv8n、YOLOv5n（同表） | 54.0、53.1、52.4、51.8、49.6 | 640 / 300 | 1.9–3.2M |
| 9 | ✓ [LightKD-SAR](https://doi.org/10.1109/JSTARS.2026.3655160) | 50.92 | 未写 | 11.43M |

**排名 4：AP，DETR 长日程**

| 排名 | 方法 | AP | 输入 / Epochs | 状态 |
|---|---|---:|---|---|
| — | [WaveFormer RT-DETR](https://doi.org/10.1109/LGRS.2025.3646494) | 61.5（阈值未写） | 未知 | abstract only |
| 1 | ✓ [MDIS-DETR](https://doi.org/10.3390/rs18132076) | 58.82 | 640 / 150，从零 | 正文 |
| — | [SAR-DETR](https://doi.org/10.1109/TAES.2026.3710962) | 58.6 | 未知 | abstract only |
| 2 | ✓ [RT-DETRv2-R18（Edge YOLO 表）](https://doi.org/10.3390/rs17132168) | 58.2 | 640 / 300 | 正文，非本方法 |
| 3 | ✓ [RT-DETR（Grad-YOLO 表）](https://doi.org/10.1109/JSTARS.2026.3692978) | 57.6 | 512，无预训练 | 正文，36.0M |
| 4 | ✓ [RT-DETR-R18（Edge YOLO 表）](https://doi.org/10.3390/rs17132168) | 57.1 | 640 / 300 | 正文 |
| — | [CARL-RTDETR](https://doi.org/10.1109/TAP.2026.3656217)、[SA-RTDETR](https://doi.org/10.1109/LGRS.2025.3632153) | 56.1、56.0 | 未知 | abstract only |
| — | [MD-DETR](https://doi.org/10.3390/rs18132076)（转引） | 55.38 | 未知 | 转引 |
| 5 | ✓ [RT-DETR 基线（MDIS 表）](https://doi.org/10.3390/rs18132076) | 54.24 | 640 / 150，从零 | 正文 |
| 6 | ✓ [Deformable DETR](https://arxiv.org/abs/2406.02833) | 52.00（MSFA 表 50.0） | 512 或 800 | 轮数说法不一，见 `deepread_detr_foundation.md` |
| 7 | ✓ [RT-DETR-R50（GSFFE 表）](https://doi.org/10.1109/JSTARS.2026.3676453) | 47.3 | — / 300，无预训练 | 正文 |
| 离群 | ✓ [CGAQ-DETR](https://doi.org/10.3390/rs17183254) | 38.6 | 640 / 7 | 只训 7 轮，不参与比较 |

**排名 5：其他指标（全集，已读正文；跨家族仅作索引，不可视为同协议）**

- **AP50**：SARATR-X-v2 L 93.9；SARLite 92.8；v2 B 92.5；Grad-YOLO 90.6；RT-DETRv2-R18 90.2；RT-DETR（Grad-YOLO 表）90.2；SUMMIT 89.9（转引）；AutoAssign 89.58；MDIS-DETR 89.31；UDCNet 89；GSFFE-Net 89.0。
- **AP75**：SARATR-X-v2 L 75.3；GSFFE-Net 71.8；v2 B 70.1；SAMBA 68.0；Grad-YOLO 与 RT-DETR 65.9；UDCNet 65.4；PGSNet 63.8；EO2SAR-Diff 63.7；MDIS-DETR 63.68；SARVLMDet 63.4。
- **APs**：SARATR-X-v2 L 67.5（日志）；EO2SAR-Diff 57.20；UDCNet 54.1；MDIS-DETR 53.43；PGSNet 53.1；DenoDet V2 51.45；MSFA 51.1。

### Inferences
- **最接近同协议的比较**只有同文内部。Grad-YOLO Table V 自跑部分（512、无预训练）：RT-DETR 57.6 ≥ YOLOv12m 56.8 ≥ YOLOv8m 56.5 ≥ YOLOv10m 55.9。Edge YOLO Table 6（640）：RT-DETRv2-R18 58.2 > RT-DETR-R18 57.1 > YOLOv10n 53.1。这补充了 `baseline_options.md` 的判断：GFL 与实时 DETR 仍无同协议比较，但实时 DETR 与 YOLO 有同表比较，且两张表都不支持“YOLO 明显强于 RT-DETR”。反例是 GSFFE Table V（RT-DETR-R50 47.3 < YOLOv8n 52.8），但其表中多行转引自 MSFA，RT-DETR 行来源需看原文说明。
- **YOLO 分成两个簇，这是本轮最重要的发现之一**。高簇（SAR-NAS、SARLite，s 规模 64–69）和低簇（Edge n 规模 50–53，Grad-YOLO m 规模 56–57）。**25.9M 的 YOLOv8m 在 512 下 56.5，低于 9.8M 的 YOLOv8-s 在 640 下 63.7**，规模大反而低 7.2 点。输入尺寸可能解释一部分，但还可能涉及 split、5 折划分或评测实现，目前无法判定。在有人用同一 split、同一 COCOeval 复跑前，**不应把 64–69 当作 YOLO 的普遍水平**。这加强了 `gaps_and_headroom.md` 关于协议人工产物的判断。
- **SARATR-X-v2 的 68.7 和 YOLO 的 69.27 是两种完全不同的路线**。前者是 iTPN-L、1200e 预训练，权重需邮件申请；后者协议未报告。把两者放在一列没有意义。
- **选基线时最该看的对照行**（推断）：DEIM-D-FINE 全集结果跑出后，应首先对照 RT-DETR 系 54–58 和 Grad-YOLO 同表 YOLO 56–57，其次才是 val 上的 DenoDet V2 56.71。

### Gaps
- 没有任何论文在同一 split、同一评测实现下同时跑 MMDet 系、YOLO 系和 DETR 系。
- Ultralytics 评测与 pycocotools 的差异在 SARDet-100K 上没有公开测量。
- 表 A 约一半的行未写明 split，排名 1、3、4 的多数名次因此只是“所在协议下的自报值”。

## Q3 矛盾：同一方法在不同论文里的数字

### Takeaway
跨论文矛盾非常普遍，**同一方法相差可达 14.7 点（YOLOv10，约 8M 规模）**。转引错误、把 test 值当 val 值、改写数字都有发生。凡是表里出现的对比行，都不应在未回到原文前使用。

### Cited Findings

| 方法 | 数字（AP，另注） | 情况 | 来源 |
|---|---|---|---|
| YOLOv10 约 8M | 54.1 / 65.1 / 68.84 | 同规模差 14.7 | [GSFFE V](https://doi.org/10.1109/JSTARS.2026.3676453)、[SARLite 1](https://www.nature.com/articles/s41598-026-49143-5/tables/1)、[SAR-NAS](https://arxiv.org/html/2509.01279v1) |
| YOLOv8 | n 51.8、n 52.8、s 63.7、m 56.5 | 规模与 AP 不单调 | [Edge 6](https://doi.org/10.3390/rs17132168)、[GSFFE V](https://doi.org/10.1109/JSTARS.2026.3676453)、[SARLite 1](https://www.nature.com/articles/s41598-026-49143-5/tables/1)、[Grad-YOLO V](https://doi.org/10.1109/JSTARS.2026.3692978) |
| RT-DETR | 47.3（R50）/ 54.24（R18）/ 57.1（R18）/ 57.6（36.0M） | 协议不同 | [GSFFE V](https://doi.org/10.1109/JSTARS.2026.3676453)、[MDIS 8](https://doi.org/10.3390/rs18132076)、[Edge 6](https://doi.org/10.3390/rs17132168)、[Grad-YOLO V](https://doi.org/10.1109/JSTARS.2026.3692978) |
| GFL R50 | 49.8 / 55.01 / 54.71 / 34.1；子集 57.31 | test 800、val 512、异常平移、7 轮、子集 | [MSFA S8](https://arxiv.org/html/2403.06534v3)、[DenoDet IV](https://arxiv.org/abs/2406.02833)、[MSPaDet 4](https://doi.org/10.3390/rs18101513)、[CGAQ 4](https://doi.org/10.3390/rs17183254)、[SM3Det 1](https://arxiv.org/abs/2412.20665) |
| Faster R-CNN R50 | 49.0 / 39.22 / 33.6；GSFFE 写 41.1（AP50、AP75 却等于 MSFA 的 82.2、52.9） | 最后一项疑似转录错误 | [MSFA S8](https://arxiv.org/html/2403.06534v3)、[DenoDet IV](https://arxiv.org/abs/2406.02833)、[CGAQ 4](https://doi.org/10.3390/rs17183254)、[GSFFE V](https://doi.org/10.1109/JSTARS.2026.3676453) |
| DETR | 31.8 / 45.73 / 7.1；GSFFE 写 50.0（85.1 / 51.7） | 后者实际是 Deformable DETR 的行 | [MSFA S8](https://arxiv.org/html/2403.06534v3)、[DenoDet IV](https://arxiv.org/abs/2406.02833)、[CGAQ 4](https://doi.org/10.3390/rs17183254)、[GSFFE V](https://doi.org/10.1109/JSTARS.2026.3676453) |
| MSFA | ConvNeXt-B 56.4；ViTP 表写 “MSFA 53.7”（实为 GFL + MSFA）；AP75 61.5，SARVLM 写 59.5 | 行名和数字被混用 | [MSFA S8–S9](https://arxiv.org/html/2403.06534v3)、[ViTP 2](https://arxiv.org/abs/2509.17562)、[SARVLM VI](https://arxiv.org/abs/2510.22665) |
| DenoDet | 55.88 / 55.4 / 55.32 / Grad-YOLO 写 0.506 | Grad-YOLO 脚注称引自原文，表值却是 50.6；正文增益也与表不符 | [DenoDet IV](https://arxiv.org/abs/2406.02833)、[MSFA README](https://github.com/zcablii/SARDet_100K)、[MSPaDet 4](https://doi.org/10.3390/rs18101513)、[Grad-YOLO V](https://doi.org/10.1109/JSTARS.2026.3692978) |
| DenoDet V2 | 56.71 / 85.86 / 61.55；MDIS 56.67 / 86.24 / 61.02；PGSNet 56.71 / 87.0 / 61.9；ViTP 56.4 | 三篇转引都与原文不同 | [V2 S7](https://arxiv.org/html/2508.09392v1)、[MDIS 8](https://doi.org/10.3390/rs18132076)、[PGSNet 1](https://ace.ewapub.com/article/view/37773.pdf)、[ViTP 2](https://arxiv.org/abs/2509.17562) |
| SARVLMDet | v3 60.2 / 92.6 / 66.9；v4 57.4 / 89.4 / 63.4 | **作者版本间下调 2.8 点**；[sota2 榜单](https://www.sota2.com/research/sota/object-detection-on-sardet-100k)仍列 60.2 | [v3](https://arxiv.org/html/2510.22665v3)、[v4](https://arxiv.org/abs/2510.22665) |
| SARATR-X | 57.3 / 57.2 | 同文两表不一致 | [Tables VII、XII](https://arxiv.org/abs/2405.09365) |
| ViTP | 59.7 / 57.9 | 论文与 README（`baseline_options.md`） | [ViTP](https://arxiv.org/abs/2509.17562) |
| MSPaDet | 56.69 / 58.33 | 同文两表；对比列逐项恒差 −0.30（`deepread_cnn_frequency.md`，本轮抽查 GFL、RepPoints、ATSS 成立） | [MSPaDet](https://doi.org/10.3390/rs18101513) |
| PGSNet 逐类 | 表 3 中 DenoDet V2 Tank 58.3、Bridge 57.1；原文 46.18、37.66 | 表 1 基线全部取自 MSFA 的 test 值，却声明“所有值基于 val” | [PGSNet](https://ace.ewapub.com/article/view/37773.pdf)、[V2 S7](https://arxiv.org/html/2508.09392v1) |
| EO2SAR-Diff 基线 | 55.88 标为 test | 等于 DenoDet val 行；同表其他方法行的末位都是 0，只有基线行是两位有效小数 | [EO2SAR 3](https://doi.org/10.3390/rs18122037)、[DenoDet IV](https://arxiv.org/abs/2406.02833) |
| BabelRS | SARATR-X-v2 Table X 把 63.3 和全集结果并列 | BabelRS 的 63.3 在 SOI-Det 上，同表 SM3Det 60.64 就是子集结果 | [v2 X](https://arxiv.org/pdf/2607.23238)、[BabelRS 2](https://arxiv.org/abs/2603.01758)、[SM3Det](https://arxiv.org/abs/2412.20665) |
| SM3Det 单模型 | GFL ConvNeXt-T 57.31，高于全集 MSFA 54.8（Faster ConvNeXt-T） | 子集效应不可归因于方法 | [SM3Det 1](https://arxiv.org/abs/2412.20665)、[MSFA S9](https://arxiv.org/html/2403.06534v3) |

**指标陷阱**
- [CGAQ-DETR 的 69.8 是 AP50](https://doi.org/10.3390/rs17183254)，AP 只有 38.6。搜索摘要容易误读。
- [MD-DETR](https://doi.org/10.1109/TGRS.2024.3489212)、[SA-DETR](https://doi.org/10.1109/JSTARS.2024.3481368) 的 88.5、88.4 是 AP50，不是 AP。
- [RingMoE 89.72](https://arxiv.org/abs/2504.03166) 是 AP50。
- [Edge YOLO Table 3–5 的 “mAP”](https://doi.org/10.3390/rs17132168) 实为 AP50，只有 Table 6 分列两种。
- 本轮未发现把 SARDet-100K 按 VOC 07/12 AP 报告的论文。[DenoDet](https://arxiv.org/abs/2406.02833)、[MSPaDet](https://doi.org/10.3390/rs18101513) 的 AP'07、AP'12 只用于 MSAR、SAR-AIRcraft。[SAR-NAS](https://arxiv.org/abs/2509.01279)、[LightKD-SAR](https://doi.org/10.1109/JSTARS.2026.3655160)、[WaveFormer](https://doi.org/10.1109/LGRS.2025.3646494) 未写明 mAP 的阈值，属于“口径未写”。

### Inferences
- 基线修正：`deepread_detr_foundation.md` 第 D 节和 `baseline_options.md` Q3 沿用了 v2 Table X，把 BabelRS 63.3 列入榜单上沿。依据表 C，**它应从全集榜移出**。这是由同表 SM3Det 行推断的，BabelRS 未单独说明子集。
- 基线修正：`deepread_cnn_frequency.md` 提到 LightKD-SAR 有 50.92 与 61.5 两个冲突数字。正文 Table III、IV 和摘要都是 50.92，而 61.5 正好是 WaveFormer RT-DETR 摘要的数字。搜索摘要可能把两篇混了，这一点只是推测。
- 基线修正：`gaps_and_headroom.md` 写 Edge YOLO 只读了摘要。本轮已读 Table 6，RT-DETR-R18 57.1 确实在表内。
- GSFFE 对比表的 Faster、DETR 行存在明显转录错误，因此其 “RT-DETR 47.3” 也不宜单独引用，除非核实是自跑结果。其正文把前 6 行和后 7 行分开说明，前者像自跑、后者像转引，这只是推断。

### Gaps
- 无法判断这些差异的成因是笔误、协议差异还是其他原因。本文只陈述算术事实，不对作者动机作判断。
- PGSNet、MSPaDet 未公开代码或日志，无法核对。

## Q4 官方划分：train / val / test 各多少，谁报告哪个？

### Takeaway
官方划分为 **train 94,493 / val 10,492 / test 11,613**。test 已于 2024-08-31 通过 issue 发布，但 README 仍写只发布 train 和 val。**文献 val 和 test 并存，且接近一半的论文没写清楚**，这是跨论文排名不可靠的直接原因之一。

### Cited Findings
- 划分与实例数：train 94,493 张 / 198,747 实例；val 10,492 / 22,703；test 11,613 / 24,023；合计 116,598 / 245,653 — [SARDet-100K Table 1](https://arxiv.org/html/2403.06534v3)。
- README：“Only Train and Val sets are released so far” — [README](https://github.com/zcablii/SARDet_100K)；[Issue #31 “Test Set Now Available!”](https://github.com/zcablii/SARDet_100K/issues/31) 提供了下载链接，日期见 `gaps_and_headroom.md`。
- 官方配置同时有 `Annotations/val.json` 和 `Annotations/test.json` — [SARDet_100k.py](https://raw.githubusercontent.com/zcablii/SARDet_100K/main/MSFA/configs/_base_/datasets/SARDet_100k.py)。
- GrokSAR 的数据集配置 `test_dataloader = val_dataloader` — `deepread_cnn_frequency.md`。

| 口径 | 论文 | 依据 |
|---|---|---|
| **test，原文明写** | [MSFA（第 12 轮权重）](https://arxiv.org/html/2403.06534v3)、[MMATR-X](https://arxiv.org/abs/2604.16952)、[EO2SAR-Diff（但基线行存疑）](https://doi.org/10.3390/rs18122037)、[SM3Det（子集 test）](https://arxiv.org/abs/2412.20665) | 正文或表题 |
| **test，由配置得知** | [SARATR-X、SARATR-X-v2（val 也指 test.json，`save_best` 开启）](https://arxiv.org/pdf/2607.23238)、[DenoDet V2 test 行](https://github.com/GrokCV/GrokSAR) | `deepread_detr_foundation.md`、`deepread_cnn_frequency.md` |
| **test，由实验推断** | [GSFFE-Net（Table III 的掩膜实验用 test，增减后等于 64.3）](https://doi.org/10.1109/JSTARS.2026.3676453) | 掩膜后 30.6，去掩膜增 33.7，合计 64.3，与 Table V 一致；基线 37.7 + 16.9 = 54.6 |
| **val，原文明写** | [Pu & Xu](https://arxiv.org/html/2501.11249v1)、[PGSNet](https://ace.ewapub.com/article/view/37773.pdf)、[SAR-LoRA-DINO README](https://github.com/Jordan-Liao/SAR-LoRA-DINO) | 正文 |
| **val，由配置得知** | [DenoDet、DenoDet V2](https://arxiv.org/html/2508.09392v1)、GFL 等 25 个对比方法 | `deepread_cnn_frequency.md` |
| **未写或未读到** | [SAR-NAS](https://arxiv.org/abs/2509.01279)、[SARLite](https://www.nature.com/articles/s41598-026-49143-5)（且消融用 5 折）、[MDIS-DETR](https://doi.org/10.3390/rs18132076)、[CGAQ-DETR](https://doi.org/10.3390/rs17183254)、[UDCNet](https://doi.org/10.1109/JSTARS.2025.3585896)、[Edge YOLO](https://doi.org/10.3390/rs17132168)、[Grad-YOLO（称用预划分集）](https://doi.org/10.1109/JSTARS.2026.3692978)、[ViTP](https://arxiv.org/abs/2509.17562)、[SAMBA](https://arxiv.org/abs/2606.31668)、[SARMAE](https://arxiv.org/abs/2512.16635)、[SARVLMDet](https://arxiv.org/abs/2510.22665)、[MSPaDet](https://doi.org/10.3390/rs18101513)、[TLTA](https://doi.org/10.3390/rs18193268)、[LightKD-SAR](https://doi.org/10.1109/JSTARS.2026.3655160) | 所读文本中未找到 |
| **非官方划分** | [SARLite 5 折](https://www.nature.com/articles/s41598-026-49143-5)、[S2M-Net 1000 张 8:2](https://www.mdpi.com/2072-4292/17/19/3347)、[SOI-Det 子集](https://arxiv.org/abs/2412.20665) | 正文 |

- MMATR-X 公开披露 SARDet-180K 预训练与 SARDet-100K test 有 160 张重叠，源于 SAR-AIRcraft-1.0，去除后 mAP 只降 0.1 — [MMATR-X §V](https://arxiv.org/abs/2604.16952)。这是本次看到的**唯一一个给出具体重叠数与影响的 SARDet-100K 论文**。

### Inferences
- 本项目的 val.json 结果**只能直接对标**“val”两行：DenoDet V2 56.71、DenoDet 55.88、GFL 55.01、Pu & Xu 54.9。其余都需说明差异。
- MMATR-X 的披露可以补进 `gaps_and_headroom.md` 的泄漏证据。它是在预训练集和 test 之间量的，只影响 0.1，但不覆盖 train 和 test 之间的重复，不能用来否定该笔记 E1 的审计设想。

### Gaps
- 未找到任何论文同时报告同一模型的 val 和 test。GrokSAR 的 56.71 与 56.39 用了不同输入和优化器，不算。
- 本轮没有逐一下载配置来确认未写 split 的论文。

## Q5 发表层级视图：顶会顶刊实际报告了什么？

### Takeaway
**顶会顶刊上 SARDet-100K 全集的最高 AP 是 CVPR 2026 的 SARMAE ViT-L 63.1，靠 307M 骨干和大规模预训练；纯检测器或模块路线在顶会上只有 AAAI 2026 的 DenoDet V2 56.71。** 全集上 64–69 的数字全部来自期刊、一般会议或待定稿件，未出现在 CCF-A 会议表中（SM3Det 的 65.20 是子集结果，不算）。SARDet-100K 上的实时 DETR 工作都发表在遥感期刊，且付费正文大多未读。

### Cited Findings

层级划分是本笔记的操作定义（第一档为 CCF-A 类计算机视觉与人工智能会议、TPAMI、TIP；第二档为遥感与雷达主流期刊），不是官方分级。

| 层级 | 论文 | 全集最高 AP | 路线 | 状态 |
|---|---|---|---|---|
| **第一档** | [SARMAE，CVPR 2026](https://arxiv.org/abs/2512.16635) | 63.1（ViT-B 57.9） | 预训练 | 本轮复核；会议信息据 `baseline_options.md` |
| | [SARATR-X，TIP 2025](https://arxiv.org/abs/2405.09365) | 57.3 | 预训练 | 本轮复核 |
| | [DenoDet V2，AAAI 2026](https://arxiv.org/html/2508.09392v1) | 56.71 | 模块 | 本轮复核 |
| | [MSFA，NeurIPS 2024](https://arxiv.org/html/2403.06534v3) | 56.4 | 预训练 | 本轮复核 |
| | [MaRS，AAAI 2026](https://doi.org/10.1609/aaai.v40i14.38153) | 55.4 | 预训练 | 转引 |
| | [RingMoE](https://arxiv.org/abs/2504.03166)、[RingMo-Agent](https://arxiv.org/abs/2507.20776)，TPAMI | 仅 AP50 89.72、53.84 | 多模态 | 本轮复核 |
| | [SM3Det，AAAI](https://arxiv.org/abs/2412.20665) | **子集** 65.20 | 多模态 | 不入全集榜 |
| | [DNOD，TMLR / ICLR Journal Track](https://iclr.cc/virtual/2026/poster/10014070) | +2.23 | DETR 频域 | abstract only |
| | [RSAR，CVPR 2025](https://arxiv.org/abs/2501.04440) | OBB | 数据集 | 另一任务 |
| **第二档** | [SARATR-X-v2](https://arxiv.org/pdf/2607.23238)（[GRSM DOI](https://doi.org/10.1109/mgrs.2026.3738381)，待确认） | 68.7 | 预训练 | 本轮复核 |
| | [GSFFE-Net，JSTARS 2026](https://doi.org/10.1109/JSTARS.2026.3676453) | 64.3 | YOLO 300e | 本轮复核 |
| | [WaveFormer RT-DETR，GRSL 2026](https://doi.org/10.1109/LGRS.2025.3646494) | 61.5 | DETR | abstract only |
| | [UDCNet，JSTARS 2025](https://doi.org/10.1109/JSTARS.2025.3585896) | 59.3 | 模块 | 本轮复核 |
| | [SAR-DETR，TAES 2026](https://doi.org/10.1109/TAES.2026.3710962) | 58.6 | DETR | abstract only |
| | [Grad-YOLO，JSTARS 2026](https://doi.org/10.1109/JSTARS.2026.3692978) | 57.6 | YOLO | 本轮复核 |
| | [SUMMIT，JAG 2025](https://doi.org/10.1016/j.jag.2025.104624) | 57.0 | 预训练 | 转引 |
| | [CARL-RTDETR，TAP 2026](https://doi.org/10.1109/TAP.2026.3656217) | 56.1 | DETR | abstract only |
| | [SA-RTDETR，GRSL 2026](https://doi.org/10.1109/LGRS.2025.3632153) | 56.0 | DETR | abstract only |
| | [DenoDet，TAES](https://arxiv.org/abs/2406.02833) | 55.88 | 模块 | 本轮复核 |
| | [MD-DETR，TGRS 2024](https://doi.org/10.1109/TGRS.2024.3489212) | 55.38（转引） | DETR | abstract only |
| | [LightKD-SAR，JSTARS 2026](https://doi.org/10.1109/JSTARS.2026.3655160) | 50.92 | 蒸馏 | 本轮复核 |
| | [SP-DETR，TGRS 2026](https://doi.org/10.1109/TGRS.2026.3729565) | 摘要无数字 | DETR | abstract only |
| **MDPI、Sci Rep** | [SARLite，Sci Rep 2026](https://www.nature.com/articles/s41598-026-49143-5/tables/1) | 67.7 | YOLO | 本轮复核 |
| | [EO2SAR-Diff，RS 2026](https://doi.org/10.3390/rs18122037) | 60.00 | 增广 | 本轮复核 |
| | [MDIS-DETR，RS 2026](https://doi.org/10.3390/rs18132076) | 58.82 | DETR | 本轮复核 |
| | [TLTA，RS 2026](https://doi.org/10.3390/rs18193268) | 56.9 | 模块 | 本轮复核 |
| | [MSPaDet，RS 2026](https://doi.org/10.3390/rs18101513) | 56.69 | 模块 | 本轮复核，存疑 |
| | [Edge YOLO，RS 2025](https://doi.org/10.3390/rs17132168) | 54.0 | YOLO | 本轮复核 |
| | [CGAQ-DETR，RS 2025](https://doi.org/10.3390/rs17183254) | 38.6 | DETR，7 轮 | 本轮复核 |
| **其他会议** | [SAR-NAS，PRCV 2025（元数据）](https://arxiv.org/abs/2509.01279) | 69.27 | YOLO | 本轮复核 |
| | [PGSNet，ACE 2026](https://ace.ewapub.com/article/view/37773.pdf) | 58.2 | 模块 | 本轮复核，存疑 |
| | [SAR-LoRA-DINO，IGARSS 2026](https://github.com/Jordan-Liao/SAR-LoRA-DINO) | 57.2 | 预训练 | README |
| | [CA-HLD，SPIE 2026](https://doi.org/10.1117/12.3130475) | 55.1 | 预训练 | abstract only |
| | [Pu & Xu，IGARSS 2025](https://arxiv.org/html/2501.11249v1) | 54.9 | 预训练 | 本轮复核 |
| **仅 arXiv（截至所见）** | [MMATR-X](https://arxiv.org/abs/2604.16952) | 61.1 | 预训练 | 本轮复核 |
| | [ViTP](https://arxiv.org/abs/2509.17562) | 59.7 | 预训练 | 本轮复核 |
| | [SAMBA](https://arxiv.org/abs/2606.31668) | 59.7 | 预训练 | 本轮复核 |
| | [SARVLMDet](https://arxiv.org/abs/2510.22665) | 57.4 | 预训练 | 本轮复核，版本下调 |

### Inferences
- 用户关心顶会顶刊时，**真正的比较对象是 56–63 这一段，而不是 69**。如果论文投 CVPR、AAAI 档，审稿人看到的上沿大概率是 SARMAE 63.1、SARATR-X 57.3、DenoDet V2 56.71。这是推断。
- 第一档论文全部是“预训练”或“模块”贡献，**没有一篇把实时 DETR 或长日程 YOLO 作为 SARDet-100K 的主要结果**。实时 DETR 路线集中在 TAES、TAP、GRSL、JSTARS 和 MDPI。这说明空位存在，但也说明顶会尚无先例可以援引其协议。
- 把 69.27 或 67.7 写成“SOTA”很可能被质疑：协议未报告或含 5 折。把它们完全忽略同样不妥，建议列为“文献报告，协议不同”。
- 第二档的实时 DETR 论文是最直接的竞争者，但数字最高的 WaveFormer 61.5 和 SAR-DETR 58.6 **都没有读到协议**。如果它们用了 500 轮或预训练，比较意义会变。优先取得这两篇全文价值最高。

### Gaps
- 分档是操作定义，未查各期刊当年分区。
- SARMAE 的 CVPR 2026、SM3Det 的 AAAI、SAR-NAS 的 PRCV 来自仓库 README 或 Semantic Scholar 元数据，未到会议官网核对。
- SARATR-X-v2 的 GRSM 发表状态只有 OpenAlex 的 DOI 记录，未访问 IEEE 页面；arXiv PDF 仍写投稿准备中。

## Q6 检索覆盖面与限制

### Takeaway
这是一次合理的单轮普查，不是穷尽。引用链覆盖较好，全文核验的主要缺口是 TGRS、TAES、TAP、GRSL 的付费正文，以及中文期刊。

### Cited Findings
- [Semantic Scholar 引用接口](https://api.semanticscholar.org/graph/v1/paper/arXiv:2403.06534/citations)：153 条，全部过了一遍标题。
- [OpenAlex](https://api.openalex.org/works?filter=cites:W4415795219)：62 条引用，其中 18 条不在 Semantic Scholar；关键词 `SARDet-100K` 共 122 条，读了前 100 条，第二页返回 429。全文过滤未成功取得结果。
- [Crossref](https://api.crossref.org/works?query=SARDet-100K) 关键词；[Europe PMC](https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=%22SARDet-100K%22&format=json) 全文检索 4 条。
- arXiv API：6 组检索式，只返回摘要命中，无法做全文检索。
- WebSearch 约 14 次，组合了 DETR、Mamba、频率、小波、自监督、基础模型、蒸馏、轻量、YOLO、域适应、扩散、few-shot 等词。
- [sota2 榜单](https://www.sota2.com/research/sota/object-detection-on-sardet-100k)只作线索，经摘要工具读取，其中 SARVLMDet 60.2 已被 v4 取代。hyper.ai 页面未返回内容。Papers with Code 未单独取到。
- 全文通道：arXiv HTML 和 PDF；`mdpi-res.com` 的 PDF（`mdpi.com` 主站返回 403）；Nature；IEEE 金色开放论文的 PDF（JSTARS 四篇成功）。SP-DETR 连接失败，SAR-DETR 被访问验证拦截，未尝试绕过。ScienceDirect 不可达。
- 本轮读到数字的正文约 28 篇，查证无 SARDet-100K 结果的正文约 10 篇。

### Inferences
- OpenAlex 与 Semantic Scholar 互相补充明显：UDCNet、PGSNet、SUMMIT 等都只出现在其中一边。单一来源的引用追踪会漏。
- 关键词命中集中在摘要，**只在实验表里用 SARDet-100K、摘要不提的论文**大概率还有遗漏。

### Gaps
- 未查 Google Scholar、CNKI、万方；中文期刊只有 OpenAlex 命中的两篇。
- OpenAlex 第 101–122 条未读。
- 未穷尽 2026 年 MDPI 和 Sci Rep 论文，也未逐条核对表 A 的代码与权重状态。
- 表 B 的所有数字都未核实，不得作为已验证结果引用。
