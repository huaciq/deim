# SARDet-100K 上 CNN / dense detector 与 signal-level 方法精读（MSFA、DenoDet、DenoDet V2 及 2025-2026 同族方法）

阅读范围说明：全文加表格已读的有 MSFA、DenoDet、DenoDet V2、MSPaDet、SARLite、S2M-Net，以及 SAR-NAS（SAR-NAS 的表格经 WebFetch 抽取，未逐字核对原始页面）。配置和代码已读的是 `zcablii/SARDet_100K`（commit b289569）和 `GrokCV/GrokSAR`（commit d41fb0e），另读了 `quest-lab-iisc/DNOD` 的配置（commit f1fb4bb）。标记为"未核实"的条目，正文无法访问，只能作为线索。

## 1. Architecture：模块插在哪里，方程和参数是什么，base detector 和 ablation 数值

### Takeaway
这一族几乎都是"GFL/FCOS + ResNet-50 + FPN，在某个 feature 层做一次正交变换，乘一个 attention mask，再反变换"。MSFA 例外，它不改网络结构，只改输入和预训练。论文描述与公开代码有几处实质差异：DenoDet 的插入位置、DenoDet V2 的 band partition 语义。所有 "phase" 都是 feature map 变换系数的相位，不是 SAR 复数回波的电磁相位。

### Cited Findings

#### MSFA（NeurIPS 2024，参考）
- 不增加 detector 模块。两部分组成：Filter Augmented Input（`Inp = concat(x, M_i^x)`，Eq.1-2）和 Multi-Stage pretrain（ImageNet 分类 → DOTA/DIOR 上整个 detector 的检测预训练 → SAR finetune，Eq.3-6）— [MSFA §4](https://arxiv.org/html/2403.06534)
- 代码中 WST 为 kymatio `Scattering2D(J=2)`，输出 81 通道，上采样回输入尺寸后与 1 通道灰度图拼接，因此 backbone 的 `in_channels` 为 82，且在 `torch.no_grad()` 下计算 — [MSFA.py](https://github.com/zcablii/SARDet_100K/blob/b289569/MSFA/msfa/models/backbones/MSFA.py)
- Table 3 / S7（Faster R-CNN + R50，均在 MSFA 框架下）的 mAP：SAR 50.2；+Canny 50.7；+HOG 50.7；+Haar 50.6；+WST 51.1；+GRE 50.6；+HOG+Haar+WST 51.1 — [MSFA Table 3 / S7](https://arxiv.org/html/2403.06534)
- Table 5（Faster R-CNN + R50）的 mAP。Raw SAR：IN 49.0；IN+DIOR Framework 49.5；IN+DOTA Backbone 49.3；IN+DOTA Framework 50.2。SAR+WST：49.2；50.1；49.6；51.1 — [MSFA Table 5](https://arxiv.org/html/2403.06534)

#### DenoDet（TAES 2024，参考）
- TransDeno 流程：2D DCT，对通道做 avg+max pooling 得到频谱统计，经两级 DeGroFC 和 sigmoid 得到 mask，做 "soft threshold"，再 IDCT。阈值定义为 `Θ = (1−σ(D_g'(δ(D_g(S_f))))) ⊙ m̃`（Eq.8-10）— [DenoDet §III-B](https://arxiv.org/html/2406.02833)
- DeGroFC 候选 group 数为 [2,4,8,16]，由 FC 产生的 K_i 经 floor/ceil 线性插值选择（Eq.12-14）— [DenoDet §III-C](https://arxiv.org/html/2406.02833)
- 论文称模块放在 "between the feature refinement neck and the detection head" — [DenoDet §III-B](https://arxiv.org/html/2406.02833)；代码实际在 FPN top-down 融合之后、`fpn_convs` 之前，只作用于 P3、P4 两层的 lateral，尺寸写死为 64×64 与 32×32 — [FrequencySpatialFPN.py](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/groksar/models/necks/FrequencySpatialFPN.py)
- host detector 按数据集切换：MSAR 用 FCOS，SAR-Aircraft 和 AIR-SARShip 用 RepPoints，SARDet-100K 用 GFL — [MSAR config](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/configs/DenoDet/DenoDet_3x_MSAR.py)、[SAR-Aircraft config](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/configs/DenoDet/DenoDet_1x_SAR-AIRcraft-1.0.py)、[SARDet config](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/configs/DenoDet/DenoDet_1x_SARDet_100k.py)
- 论文的三张 ablation 表（V、VI、VII）只覆盖 MSAR、SAR-Aircraft-1.0、AIR-SARShip-1.0，没有 SARDet-100K — [DenoDet Tables V-VII](https://arxiv.org/html/2406.02833)
- Table V（FCOS host）的 mAP(07)。MSAR：baseline 67.52；仅空间 attention 67.88；仅 DCT 67.00；DCT 加 FP64 67.37；DCT 加 attention 68.94。SAR-Aircraft：62.63；64.01；60.72；61.58；64.18。AIR-SARShip：63.66；63.70；61.91；62.06；65.91 — [DenoDet Table V](https://arxiv.org/html/2406.02833)
- Table VII 的 Deformable group 与最佳 fixed group 的 mAP(07)：MSAR 69.90 vs 69.49（group 4）；SAR-Aircraft 67.39 vs 66.10（group 2）；AIR-SARShip 66.98 vs 66.13（group 8）— [DenoDet Table VII](https://arxiv.org/html/2406.02833)

#### DenoDet V2（AAAI 2026）
- 会议信息来自仓库 README — [GrokSAR README](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/README.md)
- DFTDeno 放在 backbone，流程为 2D DFT、拆 amplitude 与 phase、attention 调制、IDFT（Eq.1-9）。phase 实际以 cos 和 sin 表示以避开角度边界不连续，并做三角恒等 alignment — [DenoDet V2 §3.1](https://arxiv.org/html/2508.09392)
- BPSA：通道方向 max+avg pooling 后按 stride (h,w) 分组，组内做 self-attention（Eq.10-14）— [V2 §3.2](https://arxiv.org/html/2508.09392)；PATE：amplitude 的 Q 配 phase 的 K、V，反之亦然（Eq.15-16）— [V2 §3.3](https://arxiv.org/html/2508.09392)
- 代码共 5 处插入：maxpool 之后的 `stem_att`，以及 4 个 ResNet stage 各自输出之后的 `att0-3`。输出直接替换 feature，没有 residual 连接；幅度依次经 `trans1(amp, sin)` 和 `trans2(amp, cos)` 调制两次；sin 和 cos 做 L2 归一化；取 IFFT 实部后接 `.relu()` — [FFTresnet.py](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/groksar/models/backbones/FFTresnet.py)
- 代码中 `partition_stride=(8,8)` 实际是 patch_num，即把频谱切成 8×8 个块，块大小为 H/8。`F.unfold` 之后 Linear 作用在"块"这一维（dim=64），每个 token 由 64 个块的同一相对位置组成 — [FFTresnet.py](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/groksar/models/backbones/FFTresnet.py)；论文则称各组 "strict inter-group information isolation" 且组内频率变化均匀 — [V2 §3.3, §4.3](https://arxiv.org/html/2508.09392)
- 论文说 positional encoding 是参数 **E** — [V2 Eq.13](https://arxiv.org/html/2508.09392)；代码是固定的 cos band_encoding，可学习版本被注释并标为 "just for debug" — [FFTresnet.py](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/groksar/models/backbones/FFTresnet.py)
- SARDet-100K 上的 ablation（全部基于 GFL），格式为 mAP / APs / APm — [V2 Tables 3-6](https://arxiv.org/html/2508.09392)

| 表 | 配置 | mAP | APs | APm |
|---|---|---|---|---|
| Table 3 | 无 split、无 alignment | 55.6 | 49.5 | 68.5 |
| Table 3 | 有 split | 56.1 | 50.4 | 68.6 |
| Table 3 | split + alignment | 56.2 | 51.4 | 68.6 |
| Table 4 | baseline | 55.0 | 49.4 | 67.3 |
| Table 4 | 仅 amplitude | 55.6 | 50.3 | 68.2 |
| Table 4 | 仅 phase | 56.2 | 51.4 | 68.6 |
| Table 4 | 两者 | 56.4 | 50.9 | 68.9 |
| Table 5 | stride 1 | 52.5 | 45.6 | 63.8 |
| Table 5 | stride 2 | 55.7 | 49.5 | 68.7 |
| Table 5 | stride 4 | 56.2 | 51.3 | 68.5 |
| Table 5 | stride 8 | 56.7 | 51.5 | 68.8 |
| Table 5 | stride 16 | 56.1 | 51.0 | 68.2 |
| Table 6 | baseline | 55.0 | 49.4 | 67.3 |
| Table 6 | No-Exchange | 56.4 | 50.9 | 68.9 |
| Table 6 | Token-Exchange | 56.7 | 51.5 | 68.8 |

#### MSPaDet（Remote Sensing 2026）
- 算法 1：FPN base（start_level=1、num_outs=5）后，对 t=0、1 两个最细层做 DTCWT（L=2，6 个方向 ±15°、±45°、±75°），PerBandNorm，PhaseReweight，FuseBands；再与 PATM（条形核 3×11、5×9）融合，最后过 PaSCA — [MSPaDet Algorithm 1](https://www.mdpi.com/2072-4292/18/10/1513)
- phase coherence 定义为 `ρ_{c,θ}(p) = |(1/|N(p)|)·Σ_{q∈N(p)} exp(i(∠W(q)−∠W(p)))|`，其中 N(p) 为 3×3 窗口，再按 `W̃ = ρ·W` 重加权；幅度做 per-band 归一化 — [MSPaDet §3.2](https://www.mdpi.com/2072-4292/18/10/1513)
- 作者明确说明 "phase-aware" 指 DTCWT 系数的 analytic phase，不是原始复数 SAR 的电磁相位 — [MSPaDet §3.2](https://www.mdpi.com/2072-4292/18/10/1513)
- SARDet-100K ablation（只报 mAP）。Table 6：baseline 53.21；仅 PhaseReweight 54.91；仅 DTCWT（FP32）53.81；仅 DTCWT（FP64）54.32；两者 55.68。Table 7：baseline 51.66；DTCWT 54.30；DTCWT+PaSCA 56.69 — [MSPaDet Tables 6-7](https://www.mdpi.com/2072-4292/18/10/1513)
- 复杂度：baseline 28.78M / 55.07G；+PaSCA 33.59M / 59.82G；+SCFRDeno 44.06M / 61.85G；完整模型 49.50M / 64.60G — [MSPaDet §4.3.4](https://www.mdpi.com/2072-4292/18/10/1513)

#### 其他
- SARLite：以 YOLO11s 为基础，stage 2-4 换成轻量 HG-block（N=5 层 depthwise，k=3，SE reduction 16），上采样换 DySample（linear + pixel shuffle，scope 0.25）。作者承认核心贡献不是新算子 — [SARLite Method](https://www.nature.com/articles/s41598-026-49143-5)
- SARLite Table 3（5-fold，mAP@50:95 / mAP@50）：YOLO11s 63.7±1.5 / 90.6±0.6；HGNet_L 57.8±1.8 / 88.3±1.4；HGNet_X 65.7±1.8 / 91.6±1.2；+Lightweight 66.1±1.1 / 91.7±0.9；+CARAFE 66.6±1.8 / 91.9±0.9；+DySample 66.7±2.0 / 92.2±0.9 — [SARLite Table 3](https://www.nature.com/articles/s41598-026-49143-5/tables/3)
- SAR-NAS：只搜 YOLOv10 backbone 每层通道宽度（0.25/0.5/0.75/1.0），SPOS 加 sandwich rule 加进化搜索 — [SAR-NAS](https://arxiv.org/html/2509.01279v1)
- S2M-Net：YOLOv11n 加 SPD-Conv、MLCA、MSDA — [S2M-Net §2](https://www.mdpi.com/2072-4292/17/19/3347)
- DNOD（TMLR 2025，ICLR 2026 Journal Track）：基于 DETR，encoder 用 Multi-Scale Fourier Mixing，decoder 用 Adaptive Deformable Fourier Neural Operator，摘要报告 SARDet-100K 上 +2.23 mAP — [ICLR 页面](https://iclr.cc/virtual/2026/poster/10014070)；配置为 R50、enc/dec 各 3 层、1200 queries、56 epochs、lr 1e-4 — [cfg_dnod.py](https://github.com/quest-lab-iisc/DNOD/blob/f1fb4bb/config/cfg_dnod.py)。它属于 transformer 族，但算子是频域的，这里仅作边界条目

### Inferences
- DenoDet Eq.5 中 `θ=(1−g)|X|`，代入 soft 函数后恒等于 `g·X`（g∈[0,1]）。所谓 "dynamic soft thresholding" 在数学上就是 sigmoid attention，并没有独立于幅度的噪声阈值，除 g 恰为 0 外也不会把系数置零。这是对同一运算的重新解释，不是新机制。
- DenoDet 与 V2 都把 mask 压成单通道后广播到全部通道（代码 `att.view(b,1,h,w).expand_as(x)`，以及 `mean + max` 后单通道 unfold）。DFT 域的乘性 mask 等价于对整幅 feature 做一次循环卷积，属于 per-image 的全局滤波，难以表达"这个区域去噪、那个区域保留"。而 speckle 是乘性且空间变化的。V2 改 phase 带来非线性，但结构上仍是全局操作。
- DenoDet V1 的参数翻倍可由代码精确复算：`SelectGroupFClayer` 在 512 输入（64²、32²）时增加约 33.5M，正好对应 65.78−32.27；256 输入（32²、16²）时增加约 2.1M，对应 MSAR 的 34.23M。这说明 "only a slight increase in parameters" 只在小输入成立，而且 FC 绑定分辨率，无法做 multi-scale。按同样公式，MSFA 仓库里 1024 输入的版本（128²、64²）约增加 535M，这是计算值，未实际构建验证。
- V2 增加的参数也可复算：10 个 CrossTransformer 分支组、dim=64，合计 0.33M，正好等于 32.60−32.27。这个结果确认了 (8,8) 确为 patch_num 语义。
- V2 论文说 stride=1 时"参数量显著增加"，与公开代码语义相反：代码中 stride=1 时 dim=1，参数几乎为零。要么论文与代码的语义不同，要么论文描述有误。
- V2 的 token 是"所有块同一相对位置"的稀疏采样，Linear 跨全部块混合。论文主张的"组间隔离"和"组内频率一致"在公开代码里并不成立。这是读代码得出的判断，未运行验证。
- 没有发现任何 SARDet-100K 上的 CNN 方法真正使用物理或散射先验。MSFA 的 WST 是数学意义上的 scattering，不是电磁散射。SARDet-100K 本身是 8-bit 幅度图，这一族也没有方法使用复数数据。

### Gaps
- DIPKD（AAAI，wavelet 噪声抑制蒸馏）正文被 bot 防护拦截，只有搜索摘要，未核实。
- LightKD-SAR 的 DOAJ 页面返回 403，只有搜索摘要，未核实架构。
- DNOD 正文位于 OpenReview 的浏览器验证之后，我没有绕过，只核实了摘要和配置。
- 未发现带 SARDet-100K 结果的 strip、large-kernel 或 CNN Mamba 检测器。Strip R-CNN 只搜到 DOTA 系结果；MaDiNet 和 SAMBA 未读正文，SAMBA 属于 foundation model，不在本族范围内。
- ASCE-SAR（YOLO11，频域-空间调制，Sci Rep 2026）只报告 OGSOD-1.0 和 HRSID，摘要未提及 SARDet-100K，故排除 — [ASCE-SAR](https://www.nature.com/articles/s41598-026-74070-w)。

## 2. Training protocol：论文正文与配置文件的核对

### Takeaway
最关键的发现是 **MSFA 表格是 test set，800 输入；DenoDet V1 和 V2 的表格是 val set，512 输入，且按 val 选 best epoch**。两边的数字不能放在一起排名。DenoDet V2 的 test-set 数字是 56.39，但用的是另一套协议（1024 输入、AdamW、stride 16）。

### Cited Findings

| 项 | MSFA（GFL/Faster） | DenoDet V1（SARDet） | DenoDet V2（表内 56.71） | DenoDet V2（test 56.39） | MSPaDet | SARLite |
|---|---|---|---|---|---|---|
| 评测集 | test，取第 12 epoch | val | val | test | 未说明 | 5-fold，或未说明 |
| 选模 | 无选择 | `save_best='auto'` | `save_best='auto'` | `save_best='auto'` | 未说明 | 未说明 |
| 输入 | 800×800 | 512×512 | 512×512 | 1024×1024 | 512×512 | 640×640 |
| Epochs | 12 | 12 | 12 | 12 | 12 | 200 |
| 优化器 | AdamW 1e-4（GFL 例外，见下） | DAdaptAdam lr=1.0 | DAdaptAdam lr=1.0 | AdamW 1e-4 | DAdaptAdam lr=1.0 | SGD 0.02 |
| Weight decay | 0.05 | 0.05 | 0.05 | 0.05 | 未单列 | 0.0005 |
| Batch | 16 | 16 | 16 | 未核 | 12 | 64 |
| 增强 | flip | flip | flip | flip | flip | flip |
| 预训练 | 自训 ImageNet 100e，再 DOTA 12e | torchvision R50 | torchvision R50 | torchvision R50 | IN | 未预训练声明 |
| 框架 | MMDet/MMPretrain，8×3090 | 论文称 MMRotate，单 3090 | MMDet，4×4090 | 同左 | MMRotate，单 3090 | 2×4090，YOLO11 框架 |

- 上表 MSFA 列来源 — [MSFA A.6、Table S12](https://arxiv.org/html/2403.06534)、[数据集配置](https://github.com/zcablii/SARDet_100K/blob/b289569/MSFA/configs/_base_/datasets/SARDet_100k.py)
- 论文明确写 "test the model on the test set with the 12th epoch checkpoint" — [MSFA A.6](https://arxiv.org/html/2403.06534)
- DenoDet V1 论文 §IV-A2 只给了 MSAR、SAR-Aircraft、AIR-SARShip 的训练细节，没有 SARDet-100K，这部分只能从配置得知 — [DenoDet §IV-A2](https://arxiv.org/html/2406.02833)、[V1 配置](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/configs/DenoDet/DenoDet_1x_SARDet_100k.py)
- GrokSAR 数据集配置中 `test_dataloader = val_dataloader`，annotation 为 `val.json`，`test.json` 段被注释 — [SARDet_100k.py](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/configs/_base_/datasets/SARDet_100k.py)；文档中的目录结构也只有 train.json 和 val.json — [datasets.md](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/docs/datasets.md)
- README 把 V2 分成 "SARDet-100K(val set)" 56.71 / 52.47G 和 "SARDet-100K(test set)" 56.39 / 0.21T 两行 — [GrokSAR README](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/README.md)；test 配置改为 1024、`partition_stride=(16,16)`、ImageNet 归一化、AdamW 1e-4 — [V2 test 配置](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/configs/DenoDetV2/DenoDetV2_1x_SARDet_100k_test.py)。论文 Table 1、S7 没有说明 56.71 是 val — [V2 Table 1](https://arxiv.org/html/2508.09392)
- V2 配置：`frozen_stages=-1`、`norm_eval=False`、SyncBN、数据集专用 mean 36.50 / std 52.12、`bgr_to_rgb=False`、`decouple=True` — [V2 配置](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/configs/DenoDetV2/DenoDetV2_1x_SARDet_100k.py)
- 基础 schedule：warmup 500 iter（1/3 起始），MultiStep [8,11]，`clip_grad max_norm=35` — [schedule_1x.py](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/configs/_base_/schedules/schedule_1x.py)
- BN 不一致：GFL 和 V2 用 SyncBN，V1 与大多数对比方法（Faster R-CNN、ATSS、FCOS、TOOD 等）用 BN；LSKNet 用 AdamW 2e-4，DETR 系用 AdamW 和更长 epoch — [GrokSAR configs 目录](https://github.com/GrokCV/GrokSAR/tree/d41fb0e/configs)
- **MSFA 的 GFL 优化器不一致**：`gfl_r50_IN_sup_sar.py` 为 SGD lr=0.01 — [IN 配置](https://github.com/zcablii/SARDet_100K/blob/b289569/MSFA/local_configs/SARDet/other_detectors/gfl_r50_IN_sup_sar.py)；`gfl_r50_dota_pretrained_sar_wavelet.py` 为 AdamW 1e-4 — [MSFA 配置](https://github.com/zcablii/SARDet_100K/blob/b289569/MSFA/local_configs/SARDet/other_detectors/gfl_r50_dota_pretrained_sar_wavelet.py)；DOTA 预训练阶段也是 SGD 0.01 — [pretrain 配置](https://github.com/zcablii/SARDet_100K/blob/b289569/MSFA/local_configs/SARDet/other_detectors/pretrain_gfl_r50_sar_wavelet.py)。论文 Table S12 却写 finetune 统一 AdamW 1e-4 — [MSFA Table S12](https://arxiv.org/html/2403.06534)。其余 Faster R-CNN、Cascade、Grid、RetinaNet 配置的两侧优化器一致（均为 AdamW）
- MSFA 仓库 README 在 GFL 组内多列一行 DenoDet：55.4 (+5.6)，@50 84.7，@75 58.3，s 49.5，m 67.6，l 63.2 — [MSFA README](https://github.com/zcablii/SARDet_100K/blob/b289569/README.md)。这一行对应的配置是 1024 输入、AdamW 1e-4、torchvision 预训练（未用 MSFA）、`frozen_stages=1`、FPN 加 GN，与 49.8 的基线配置不同 — [DenoDet 配置](https://github.com/zcablii/SARDet_100K/blob/b289569/MSFA/local_configs/DeonDet/gfl_r50_denodet_sardet.py)
- SARLite 的 5-fold：把数据集切成 5 个不重叠 fold，每次留一个做验证 — [SARLite Implementation detail](https://www.nature.com/articles/s41598-026-49143-5)。这不是官方 train/val/test 协议
- S2M-Net 的 SARDet-100K 实验只选了 1000 张纯 ship 图像，按 8:2 划分；SGD、200 epochs、640 输入、mosaic — [S2M-Net §3.1-3.2](https://www.mdpi.com/2072-4292/17/19/3347)
- SAR-NAS 页面未报告最终模型的输入尺寸、优化器、划分和 seed，只提到 supernet 500 epochs、batch 256 和 validation dataset — [SAR-NAS](https://arxiv.org/html/2509.01279v1)

### Inferences
- 这些仓库的配置中都未发现显式固定的 seed（grep `randomness`、`seed` 只命中 sampler hook），也没有多 seed 脚本。
- V2 配置中 `batch_size=16` 是 per-GPU。如果按论文所说用 4 卡，全局 batch 是 64；如果论文的 16 指全局，则与配置不符。V1 标称单卡。因此 V1 与 V2 的对比可能存在 batch 差异，论文无法判定。
- V2 论文称 "all experiments employed MMDetection with four RTX 4090"，但其 25 个对比方法的数字与 V1 Table IV（MMRotate、单 3090）逐项相同，所以对比方法大概率是复用，不是重跑。
- 公开配置不一定就是实际产出表格的配置。例如 MSFA 的 FCOS 配置 `_base_` 里写的是 `datasets/dota.py`。因此上述 GFL 优化器差异是"公开证据上的混淆"，不等于确证实际实验如此。

### Gaps
- V1 的 SARDet-100K 表格究竟用 val 还是 test，论文未说，只能由配置和 README（V2 明确 val）推断。
- V1 的 test-set 数字只有 MSFA README 那一行（55.4），未见 GrokSAR 官方的 V1 test 数字。
- 未核实 MSPaDet、SAR-NAS 的代码（MSPaDet 代码"按需索取"）。

## 3. Results：表格数值

### Takeaway
官方 val 协议下三者依次为 GFL 55.01、V1 55.88、V2 56.71。test 协议下 V2 56.39，只与 MSFA 的 Faster R-CNN + ConvNeXt-B（56.4）打平，而且 V2 的输入更大。MSPaDet 自报 56.69，但其对比列存在异常，不宜作为证据。YOLO 系报告的 63-69 mAP 与 mmdet 系不可比。

### Cited Findings

**DenoDet V1 Table IV / DenoDet V2 Table S7（val，512，12e，IN；V2 的表头未注明 val）**

| 方法 | FLOPs | Params | mAP | AP50 | AP75 | APs | APm | APl | Ship | Aircraft | Car | Tank | Bridge | Harbor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GFL | 52.36G | 32.27M | 55.01 | 85.16 | 58.87 | 49.44 | 67.29 | 60.45 | 63.92 | 57.63 | 62.29 | 44.80 | 36.41 | 65.04 |
| ATSS | 51.57G | 32.13M | 54.95 | 87.60 | 58.25 | 49.89 | 67.94 | 58.97 | 61.53 | 55.94 | 61.77 | 46.20 | 37.22 | 67.48 |
| TOOD | 50.52G | 30.03M | 54.65 | 86.88 | 58.41 | 50.20 | 66.72 | 58.60 | 62.28 | 55.61 | 62.53 | 45.96 | 36.64 | 65.24 |
| AutoAssign | 51.83G | 36.26M | 53.95 | 89.58 | 55.96 | 50.14 | 63.40 | 54.73 | 62.03 | 55.70 | 61.69 | 48.55 | 38.25 | 57.45 |
| Faster R-CNN | 63.2G | 41.37M | 39.22 | 70.04 | 39.87 | 32.55 | 47.23 | 42.02 | 50.45 | 50.36 | 57.82 | 24.90 | 18.69 | 33.11 |
| Cascade R-CNN | 90.99G | 69.17M | 53.55 | 87.33 | 56.81 | 49.09 | 62.89 | 48.68 | 66.99 | 56.43 | 63.25 | 44.35 | 36.89 | 53.81 |
| DenoDet | 52.69G | 65.78M | 55.88 | 85.81 | 60.16 | 50.63 | 68.47 | 60.96 | 64.91 | 57.36 | 63.66 | 45.79 | 36.39 | 67.17 |
| DenoDet V2 | 52.47G | 32.60M | 56.71 | 85.86 | 61.55 | 51.45 | 68.75 | 61.34 | 64.85 | 58.08 | 63.26 | 46.18 | 37.66 | 70.22 |

来源 — [DenoDet Table IV](https://arxiv.org/html/2406.02833)、[DenoDet V2 Table S7](https://arxiv.org/html/2508.09392)

**MSFA Table S8 / S9（test，800，12e，单位 mAP；INP→MSFA）**

| 模型 | INP mAP | MSFA mAP | 增益 | INP AP50 | MSFA AP50 | INP AP75 | MSFA AP75 | MSFA APs | MSFA APm | MSFA APl |
|---|---|---|---|---|---|---|---|---|---|---|
| Faster R50 | 49.0 | 51.1 | +2.1 | 82.2 | 83.9 | 52.9 | 54.7 | 45.2 | 62.3 | 57.5 |
| Cascade R50 | 51.1 | 53.9 | +2.8 | 81.9 | 83.4 | 55.8 | 59.8 | 47.2 | 66.1 | 63.2 |
| Grid R50 | 48.8 | 51.5 | +2.7 | 79.1 | 81.7 | 52.9 | 56.3 | 45.1 | 64.1 | 60.0 |
| RetinaNet | 47.4 | 49.0 | +1.6 | 79.3 | 80.1 | 49.7 | 52.6 | 41.3 | 61.1 | 59.4 |
| GFL | 49.8 | 53.7 | +3.9 | 80.9 | 84.2 | 53.3 | 57.8 | 47.8 | 66.2 | 59.5 |
| FCOS | 46.5 | 48.5 | +2.0 | 80.9 | 82.1 | 49.0 | 51.4 | 42.9 | 60.4 | 56.0 |
| DETR | 31.8 | 47.2 | +15.4 | 62.3 | 77.5 | 30.0 | 49.8 | 37.9 | 62.9 | 58.2 |
| Deformable DETR | 50.0 | 51.3 | +1.3 | 85.1 | 85.3 | 51.7 | 54.0 | 44.9 | 65.6 | 61.7 |
| Sparse R-CNN | 38.1 | 41.4 | +3.3 | 68.8 | 74.1 | 38.8 | 41.8 | 33.6 | 53.9 | 53.4 |
| DAB-DETR | 45.9 | 48.2 | +2.3 | 79.0 | 81.1 | 47.9 | 51.0 | 41.2 | 63.1 | 55.4 |
| Faster R101 | 51.2 | 52.0 | +0.8 | 84.1 | 84.6 | 55.6 | 56.6 | 46.6 | 63.4 | 57.7 |
| Faster R152 | 51.9 | 52.4 | +0.5 | 85.2 | 85.4 | 55.9 | 57.2 | 47.4 | 63.3 | 58.7 |
| Faster ConvNeXt-T | 53.2 | 54.8 | +1.6 | 86.3 | 87.1 | 58.1 | 59.8 | 48.8 | 66.7 | 62.1 |
| Faster ConvNeXt-S | 54.2 | 55.4 | +1.2 | 87.8 | 87.6 | 59.2 | 60.7 | 50.1 | 67.1 | 61.3 |
| Faster ConvNeXt-B | 55.1 | 56.4 | +1.3 | 87.8 | 88.2 | 59.5 | 61.5 | 51.1 | 68.3 | 62.4 |
| Faster VAN-B | 53.5 | 55.1 | +1.6 | 86.8 | 87.7 | 58.0 | 60.2 | 48.8 | 67.3 | 62.2 |
| Faster Swin-T | 48.4 | 50.2 | +1.8 | 83.5 | 84.1 | 50.8 | 53.9 | 44.1 | 61.3 | 58.8 |
| Faster Swin-B | 53.8 | 55.7 | +1.9 | 87.8 | 87.8 | 59.0 | 61.4 | 50.5 | 66.5 | 62.5 |

来源 — [MSFA Tables S8-S9](https://arxiv.org/html/2403.06534)

**其他**
- MSFA Table S11：Faster R-CNN 直接 finetune 36e 得 52.8；DOTA 12e 加 finetune 24e 得 54.5 — [MSFA Table S11](https://arxiv.org/html/2403.06534)
- MSFA Table S13（ConvNeXt-B MSFA，test）的 AP：ship 66.9，aircraft 45.5，car 65.5，tank 45.4，bridge 44.1，harbor 71.1 — [MSFA Table S13](https://arxiv.org/html/2403.06534)
- MSPaDet Table 4 自报：64.60G，49.50M，mAP 56.69，AP50 86.32，AP75 61.17，APs 50.80，APm 68.30，APl 62.50，Ship 65.94，Aircraft 58.33，Car 64.32，Tank 45.13，Bridge 37.81，Harbor 68.50；Table 6 中 SARDet-100K 的 FPS 为 101.9 — [MSPaDet Tables 4, 6](https://www.mdpi.com/2072-4292/18/10/1513)
- SARLite Table 1（mAP@50:95 / mAP@50，GFLOPs，Params）：YOLOv8-s 63.7 / 90.3（23.4，9.8M）；YOLOv10-s 65.1 / 90.9（24.5，8.0M）；YOLO11-s 64.0 / 91.2（21.3，9.4M）；Proposed 67.7 / 92.8（20.3，7.8M）— [SARLite Table 1](https://www.nature.com/articles/s41598-026-49143-5/tables/1)
- SARLite 的每类 AP50：tank 98.8，car 98.6，aircraft 96.7，bridge 92.4，ship 86.4，harbor 82.6 — [SARLite Results](https://www.nature.com/articles/s41598-026-49143-5)
- SAR-NAS Table 2（mAP，FLOPs，Params）：YOLOv10-N 59.47（6.53G，2.27M）；SAR-NAS-N 60.18（5.96G，1.97M）；YOLOv10-S 68.84（21.43G，7.22M）；SAR-NAS-S 69.27（20.17G，7.07M）。未报告 AP50、AP75、尺度和每类 AP — [SAR-NAS](https://arxiv.org/html/2509.01279v1)
- S2M-Net（1000 张纯 ship 子集）mAP50：baseline 0.874，完整模型 0.883；同表 YOLOv8s 为 0.891 — [S2M-Net Tables 3-4](https://www.mdpi.com/2072-4292/17/19/3347)
- 第三方论文 ViTP 的 Table 4 列出 GFL 49.8、MSFA 53.7、DenoDet 55.4、DenoDetv2 56.4、SARATR-X 57.3、ViTP 59.7，但未说明对比数字来源和 split — [ViTP Table 4](https://arxiv.org/html/2509.17562v2)
- LightKD-SAR 搜索摘要有 50.92% / 15.7G / 11.43M 和 61.5% 两个相互冲突的数字，未核实 — [DOAJ 条目](https://doaj.org/article/32055ce874b248ab8beee9fb3f63cfdd)

### Inferences
- ViTP 的 GFL 49.8、MSFA 53.7、DenoDet 55.4 与 MSFA README 完全一致，DenoDetv2 56.4 与 GrokSAR test 行 56.39 一致，因此这些很可能都是 test 数字的转引。但这几行的输入和优化器并不相同（800 SGD、800 AdamW、1024 AdamW、1024 AdamW）。
- 目前能拼出的最接近同口径的 test 序列是：GFL INP 49.8 → GFL MSFA 53.7 → GFL+DenoDet 55.4 → GFL+V2 56.39，以及 Faster+ConvNeXt-B MSFA 56.4。序列内每一步都混入了协议变化，不能直接读作模块增益。
- V1 和 V2 的 "No. 1" 都是针对 val、512、12e 下的 25 个方法成立的。这个排名在 test 口径下并未被证明。

### Gaps
- V2 test-set 版本没有公开逐类 AP 和 AP50/AP75（README 只列 mAP）。
- 除 MSFA 的 Table S13 外，MSFA 各行没有每类 AP。
- MSFA、V1（SARDet）、V2 均未在 SARDet-100K 上报告 FPS。

## 4. Weaknesses：方法学弱点

### Takeaway
核心问题有四个：单次运行、无方差；选模与报告用同一 val；host detector 按数据集挑选；数据集内的模块增益（+0.3 到 +1.7）小于换 backbone、延长训练、改优化器或改输入带来的差异。另外 MSPaDet 和 SARLite 的对比数字存在可计算的异常。

### Cited Findings

#### 通用
- MSFA、DenoDet、V2、MSPaDet 的 SARDet-100K 表格都没有报告 std 或多 seed；有 cross-validation 和显著性检验的是 SARLite — [SARLite](https://www.nature.com/articles/s41598-026-49143-5)，S2M-Net 有 5-fold 但表内不报方差 — [S2M-Net §3.2](https://www.mdpi.com/2072-4292/17/19/3347)
- SARLite 的 fold 间 std 为 1.1-2.0 mAP — [SARLite Table 3](https://www.nature.com/articles/s41598-026-49143-5/tables/3)。这是跨 fold 方差，不是 seed 方差，但量级提示 0.3-1 的增益需要统计检验
- V1、V2 没有注入 speckle 的鲁棒性曲线，去噪证据是 Grad-CAM 或 Eigen-CAM 可视化 — [DenoDet §V](https://arxiv.org/html/2406.02833)、[V2 §4.4](https://arxiv.org/html/2508.09392)

#### MSFA
- WST 单独在纯 ImageNet 预训练下只有 +0.2（49.0→49.2），只有与 DOTA framework 预训练组合时才到 +0.9（50.2→51.1）— [MSFA Table 5](https://arxiv.org/html/2403.06534)
- Table 4 的 PCC 与 mAP 几乎不对应：Hog 0.995、Canny 0.992、Haar 0.990、GRE 0.984，对应 mAP 只在 50.6-50.7 — [MSFA Tables 3-4](https://arxiv.org/html/2403.06534)
- backbone 越大增益越小：R50 +2.1、R101 +0.8、R152 +0.5 — [MSFA Table S9](https://arxiv.org/html/2403.06534)
- 作者列出的失败类别是小目标（Tank，平均面积 461 px）、大长宽比（Bridge）、高外观变化（Airplane）— [MSFA A.8](https://arxiv.org/html/2403.06534)
- Table S13 的 Average 行 @m 写 0.733，而 Table S9 为 68.3，六类平均也是 0.683，属于笔误 — [MSFA Tables S9, S13](https://arxiv.org/html/2403.06534)
- SOTA 对比（Table 6）中多数 SAR 方法标为不开源，无法同协议重跑，只能取文献数字 — [MSFA Table 6](https://arxiv.org/html/2403.06534)
- Table S11 的 Total Iterations 写 17.7k（36 epochs）和 16.1k（12+24 epochs）— [MSFA Table S11](https://arxiv.org/html/2403.06534)。按 Table 1 的 94,493 张训练图、batch 16 计算，36 epochs 约为 212k iter，表内数字不吻合（两行比值约 0.91 则与计算相符，可能是单位或换算错误）

#### DenoDet V1
- 没有 SARDet-100K ablation，模块对该数据集的贡献只能从 GFL 55.01→55.88 这一对数字读出 — [DenoDet Tables IV-VII](https://arxiv.org/html/2406.02833)
- 增益并不总为正：Table V 中空间 attention 在 AIR-SARShip 和 SAR-Aircraft 的 mAP(12) 低于 baseline（63.50 vs 64.25，60.40 vs 63.76）；Table VI 中 DCT 不加 transposition 在 AIR-SARShip 为 60.30 vs 63.66 — [DenoDet Tables V-VI](https://arxiv.org/html/2406.02833)
- 正文强调 MSAR 上提升 2.5%，表内实际为 69.90−67.52=2.38（AP12 为 1.36），且 AutoAssign 已达 69.41 — [DenoDet Table II](https://arxiv.org/html/2406.02833)
- 表间不一致：SAR-Aircraft 上 Table III 为 68.60（RepPoints host），Table VII "Deformable" 为 67.39，Table V 为 64.18（FCOS host）— [DenoDet Tables III, V, VII](https://arxiv.org/html/2406.02833)；论文未说明 host 变化，host 差异可从配置得知 — [SAR-Aircraft config](https://github.com/GrokCV/GrokSAR/blob/d41fb0e/configs/DenoDet/DenoDet_1x_SAR-AIRcraft-1.0.py)
- 表内笔误：Table III GFL 的 FLOPs 写成 32.27G（实为其 Params 数值）；Table IV ConvNeXtV2 的 Params 写成 0.11G — [DenoDet Tables III-IV](https://arxiv.org/html/2406.02833)

#### DenoDet V2
- stride 超参在报告集上选择，且范围大：52.5 到 56.7，相邻取值差 0.5-0.6 — [V2 Table 5](https://arxiv.org/html/2508.09392)
- token exchange 只带来 +0.3（56.4→56.7），APm 还下降 0.1 — [V2 Table 6](https://arxiv.org/html/2508.09392)
- phase-only 的 APs（51.4）高于 amplitude+phase（50.9），这与正文 "do not conflict" 的表述并不完全一致 — [V2 Table 4](https://arxiv.org/html/2508.09392)
- "Phase 比 amplitude 更鲁棒" 的证据只有单次运行 56.2 vs 55.6 的 0.6 差距 — [V2 §4.3](https://arxiv.org/html/2508.09392)
- 摘要的 "reducing the model complexity by half" 只指 Params（65.78M→32.60M），FLOPs 只从 52.69G 降到 52.47G — [V2 Table 1](https://arxiv.org/html/2508.09392)
- 文本错误：Eq.4 多出 i、频谱对称中心写成 (2/H, 2/W)，Table 1 ConvNeXtV2 的 #P 写成 110.0G — [V2 §3](https://arxiv.org/html/2508.09392)
- SAR-Aircraft 和 AIR-SARShip 的 AP 对比中，LSKNet 在 AIR-SARShip 已到 71.66，V1 为 72.42、V2 为 73.98 — [V2 Table 2](https://arxiv.org/html/2508.09392)

#### MSPaDet（对比列存在异常）
- 对其 Table 4 与 DenoDet Table IV 做逐项差值：GFL、RepPoints、ATSS、PVT-T、TOOD、DDOD、YOLOF、YOLOX、ConvNeXt、ConvNeXtV2、DAB-DETR 共 11 行的全部 12 个指标都恰好为 −0.30；FCOS 行差值全为 0 — [MSPaDet Table 4](https://www.mdpi.com/2072-4292/18/10/1513)、[DenoDet Table IV](https://arxiv.org/html/2406.02833)
- 其 MSAR 表和 SAR-Aircraft 表也有同样模式：FCOS 的 MSAR 指标抽查均为 −1.30（67.52→66.22，89.17→87.87，43.72→42.42）；GFL 的 SAR-Aircraft 指标抽查均为 −5.50（66.90→61.40，59.90→54.40，87.34→81.84）— [MSPaDet Tables 2-3](https://www.mdpi.com/2072-4292/18/10/1513)、[DenoDet Tables II-III](https://arxiv.org/html/2406.02833)
- 其 Table 6 的 FPS 列（230.4、230.1、225.0、220.3、215.0；167.6、167.4、167.3、152.3、166.4）与 DenoDet Table V 的前两组完全相同 — [MSPaDet Table 6](https://www.mdpi.com/2072-4292/18/10/1513)、[DenoDet Table V](https://arxiv.org/html/2406.02833)
- 内部不一致：SARDet-100K 的 baseline 在 Table 6 为 53.21、Table 7 为 51.66、Table 4 的 FCOS 为 52.52；完整模型 Table 4 和 7 为 56.69，Table 5 为 0.5833；baseline 28.78M 与 FCOS 32.13M 不符；数据集规模误写为 16,598 张 — [MSPaDet](https://www.mdpi.com/2072-4292/18/10/1513)
- 增益表述不一：正文 "surpasses DiffDet4SAR by 0.9"，结论 "0.57% gain"（对 MSDFEN）— [MSPaDet §4.2.3, §6](https://www.mdpi.com/2072-4292/18/10/1513)
- 表中未包含 DenoDet V2，代码也只"按需提供"— [MSPaDet](https://www.mdpi.com/2072-4292/18/10/1513)

#### SARLite、SAR-NAS、S2M-Net
- SARLite Table 1 的 ResNet-50、ResNet-101、ConvNeXt-T、Swin-T 的 mAP@50 与 MSFA Table S9 的 MSFA 行完全相同（83.9、84.6、87.1、84.1），mAP@50:95 分别为 53.2、52.8、56.4、52.0 — [SARLite Table 1](https://www.nature.com/articles/s41598-026-49143-5/tables/1)、[MSFA Table S9](https://arxiv.org/html/2403.06534)
- SARLite 表 1 与表 3 的同一模型数字不同（YOLO11-s 64.0 vs 63.7；Proposed 67.7 vs 66.7）；正文说 DySample 与 nearest 差异不显著（p=0.23），结论却说每个组件都有显著贡献（p<0.001）— [SARLite](https://www.nature.com/articles/s41598-026-49143-5)
- SAR-NAS 的 Table 1 对比数字与 DenoDet Table IV 相同，而 YOLOv10 系协议未报告 — [SAR-NAS](https://arxiv.org/html/2509.01279v1)、[DenoDet Table IV](https://arxiv.org/html/2406.02833)
- S2M-Net 的 SARDet 子集验证集约 200 张图，八组 ablation 只在 0.856-0.883 间波动，且不单调；同表 YOLOv8s 0.891 高于本方法；正文 LR 写 1e-3，Table 2 写 0.01 — [S2M-Net](https://www.mdpi.com/2072-4292/17/19/3347)

### Inferences

**增益量级对比（在论文内部数字上计算）**

| 变化 | 增益（mAP） | 依据 |
|---|---|---|
| WST（仅 ImageNet 预训练） | +0.2 | MSFA Table 5 |
| PATE token exchange | +0.3 | V2 Table 6 |
| WST（在 DOTA 预训练上） | +0.9 | MSFA Table 5 |
| DenoDet V1 vs GFL | +0.87 | V1 Table IV |
| DOTA Framework 预训练（Raw SAR） | +1.2 | MSFA Table 5 |
| DenoDet V2 vs GFL | +1.70 | V2 Table S7 |
| MSFA 全量（R50，Faster） | +2.1 | MSFA Table S9 |
| 1x 改 3x（Faster，IN） | +3.8 | 52.8 vs 49.0，Table S11 与 S9 |
| R50 换 ConvNeXt-T（Faster，INP，参数相近） | +4.2 | 53.2 vs 49.0，Table S9 |
| V2 stride 取值范围 | 4.2 | 52.5 到 56.7，V2 Table 5 |
| R50 换 ConvNeXt-B（Faster，INP） | +6.1 | 55.1 vs 49.0，Table S9 |
| 同为 GFL R50，协议差异（val 512 DAdaptAdam vs test 800 SGD） | 5.2 | 55.01 vs 49.8 |
| 同为 Faster R50，协议差异（val 512 DAdaptAdam vs test 800 AdamW） | −9.8 | 39.22 vs 49.0 |

- 上表结论明确：**模块增益明显小于预训练、backbone、schedule 和优化器-检测器组合的差异**。Faster R-CNN 在 GrokSAR 表里 39.22，比自家其他方法低 10 个点以上，而 MSFA 同类配置为 49.0，说明 DAdaptAdam lr=1.0 / 512 的统一协议并不对所有方法公平。方法排名高度依赖协议。
- V1 和 V2 选择的 GFL 恰好是其协议下最高的 one-stage host；在这个 host 上增益是否能迁移到 ATSS、TOOD 等，SARDet-100K 上没有证据。V1 在 MSAR 和 aircraft 上换 host，进一步说明"plug-and-play"的泛化并未在同一数据集上验证。
- MSFA 的 GFL +3.9 是常规检测器里最大的增益，同时也是公开配置中两侧优化器不同的唯一一行（SGD 0.01 vs AdamW 1e-4），因此需要打折。其余常规检测器的增益约为 +1.6 到 +2.8。MSFA README 里 DenoDet 的 "+5.6" 同时变了输入（1024 vs 800）、优化器和 neck 归一化，不能当作模块增益。
- MSPaDet 的逐项恒定 −0.30、−1.30、−5.50 不可能来自独立重训（重训不会让 12 个指标平移同一常数）。我无法判断原因，只能陈述这个算术事实。在作者提供原始日志之前，建议不把其对比列和增益当作证据，也不把 56.69 当作可信的 SOTA 基线。
- SARLite Table 1 四个 backbone 的 mAP 恰好等于 MSFA Table S9 的 "MSFA mAP + 括号增益"：51.1+2.1=53.2，52.0+0.8=52.8，54.8+1.6=56.4，50.2+1.8=52.0。这很像把括号增益重复加了一次的转录错误，且这些行来自 Faster R-CNN 800 / 12e，而不是 YOLO 框架下的实验。
- YOLO 系 63-69 mAP 不能解读为"轻量 YOLO 比 mmdet 系高 10 点"。协议完全不同（200 epochs、mosaic、其他划分或 CV），且每类模式明显反转：SARLite 的 tank AP50 98.8、aircraft 96.7、harbor 82.6，而 ConvNeXt-B MSFA 为 tank 76.6、aircraft 75.3、harbor 97.9（Table S13）。如此大的反转值得单独查明：可能涉及划分、评测实现或标签映射，目前无法判定。但也不能排除长训练和强增强确实有大幅增益，这一点需要在同一 split、同一 COCOeval 下实验验证。
- Aircraft AP 差异也值得注意：V2 在 val 为 58.08，MSFA ConvNeXt-B 在 test 为 45.5。虽然检测器不同，但差距过大，提示 val 与 test 的 aircraft 难度可能不同（Table 1 显示 SAR-AIRcraft 的 val/test 实例数分别为 4,631 / 5,996）。这是假设，未验证。

### Gaps
- 没有任何一篇在相同 split、相同输入和多 seed 下同时比较 MSFA、DenoDet、DenoDet V2。
- 没有找到这些方法的第三方独立复现。

## 5. Where the gain comes from：AP75、AP50 还是小目标

### Takeaway
频域模块的增益主要来自 **高 IoU 定位（AP75）和小/中目标，不是召回式的 AP50**。AP50 几乎不变。更重要的是，**DenoDet V2 增益的一半来自 Harbor 一个类别**。Bridge、Tank 仍然最低。

### Cited Findings
- V1 − GFL（val）：mAP +0.87，AP50 +0.65，AP75 +1.29，APs +1.19，APm +1.18，APl +0.51；Ship +0.99，Aircraft −0.27，Car +1.37，Tank +0.99，Bridge −0.02，Harbor +2.13（由表计算）— [DenoDet Table IV](https://arxiv.org/html/2406.02833)
- V2 − GFL（val）：mAP +1.70，AP50 +0.70，AP75 +2.68，APs +2.01，APm +1.46，APl +0.89；Ship +0.93，Aircraft +0.45，Car +0.97，Tank +1.38，Bridge +1.25，Harbor +5.18（由表计算）— [V2 Table S7](https://arxiv.org/html/2508.09392)
- V2 − V1（val）：mAP +0.83，AP50 +0.05，AP75 +1.39，APs +0.82，APm +0.28，APl +0.38；Ship −0.06，Aircraft +0.72，Car −0.40，Tank +0.39，Bridge +1.27，Harbor +3.05（由表计算）— [V2 Table S7](https://arxiv.org/html/2508.09392)
- 每类 AP 的平均与 mAP 一致（GFL 55.015，V1 55.88，V2 56.708），可确认这些行内部自洽 — [V2 Table S7](https://arxiv.org/html/2508.09392)
- WST（Table S7，相对 SAR as RGB）：mAP +0.9，AP50 +0.9，AP75 −0.1，APs +0.4，APm +0.7，APl −1.1 — [MSFA Table S7](https://arxiv.org/html/2403.06534)
- MSFA 全量（由表计算）：Faster R50 的 AP50 +1.7、AP75 +1.8、APs +1.7、APm +1.7、APl +2.5；Cascade 的 AP50 +1.5、AP75 +4.0；ConvNeXt-B 的 AP50 +0.4、AP75 +2.0、APs +2.2、APm +1.4、APl +1.3 — [MSFA Tables S8-S9](https://arxiv.org/html/2403.06534)
- AutoAssign 的 AP50（89.58）、Deformable DETR（88.77）、ATSS（87.60）都高于 V2（85.86），而 AP75 明显低于 V2（55.96、54.03、58.25 vs 61.55）— [V2 Table S7](https://arxiv.org/html/2508.09392)

### Inferences
- V2 的增益中，Harbor 贡献 5.18/6 = 0.86 个 mAP 点，占 50.8%；去掉 Harbor 后，其余五类平均只有 +1.00。V1 中 Harbor 占 40.8%，其余五类平均只有 +0.61。论文叙述的重点是"受噪声影响的小、中目标"，并未解释增益为何集中在 Harbor 这一个类别，因此去噪叙事没有被这个类别分布很好地支持。
- 同时 APs 确有 +2.01，所以小目标的改善也存在；但 Aircraft 只有 +0.45，因此把增益归因于 "speckle 抑制" 需要更多证据。
- 频域模块是"定位质量提升型"：V2 的 AP75 增益是 AP50 的约 3.8 倍。MSFA 的 detection-level 预训练在大模型上也是如此。两者并不互相排斥，但都没有解决 GFL 系 AP50 只有 85 左右、落后于 AutoAssign 89.6 的问题。在 AP50 上，选择 assigner 的影响可能大于频域模块。
- WST 输入的增益只来自 AP50 和 APm，AP75 和 APl 反而下降，说明它是"召回型"的微弱增益，并不提升定位。
- 持续偏低的类别：Bridge（V2 37.66，MSFA ConvNeXt-B 44.1）、Tank（V2 46.18，MSFA ConvNeXt-B 45.4），以及 test 上的 Aircraft（45.5）。Bridge 在更大输入和 ConvNeXt-B 下明显更高，而频域模块只增加约 1.25，提示大长宽比目标对分辨率和 backbone 的敏感度高于对去噪的敏感度。这是跨协议比较，只作假设。
- 对后续选基线的启发：（1）任何 SARDet-100K 对比都要先固定 split，官方 test 与 val 差异可达 0.3 以上（V2 自身 val 56.71 与 test 56.39 虽属不同协议）；（2）模块增益不到 2 个点时，必须多 seed 才能宣称有效；（3）报告时应同时给出每类 AP，并检查增益是否集中在 Harbor；（4）在考虑频域模块之前，预训练、schedule 和 backbone 的提升空间更大。

### Gaps
- V1、V2 未公开 test-set 的 AP75、尺度和每类分解，上述分解只在 val 上成立。
- 所有论文都没有做 TIDE 式的错误分解（定位、分类、漏检、背景），AP75 增益究竟来自定位改善还是打分改善，目前无法区分。
- Harbor 增益的原因（例如 Harbor 主要来自 OGSOD 子集时的数据集效应）没有论文分析，也没有按 source dataset 拆分的结果。
