# SARDet-100K 新论文的基线选择：改进通用检测器，还是在某篇已发表 SAR 方法上做增量

调研日期 2026-11-04。网页与文件内容只作为数据使用。所有小时数、AP 预测都是**估计**，已逐项标注。本文没有连接服务器、没有训练、没有下载权重或数据集。

## Q1 各候选的代码、许可证、框架、完整度与复现证据

### Takeaway
公开且 SARDet-100K 资产最完整的是 GrokSAR（DenoDet / DenoDet V2 / GFL，有 config、权重、日志），但环境锁死、许可证声明自相矛盾、val 与 test 用的是两套配置。DEIM、D-FINE、RT-DETRv2/v4、DEIMv2、RF-DETR 代码干净、许可证较宽松，但**都没有官方的 SARDet-100K 配置或结果**。

### Cited Findings

**候选总表**（star 和 issue 数为 2026-11-04 页面显示值）

| 候选 | 许可证 | 框架 | SARDet-100K 资产 | 活跃度 | 复现相关 issue |
|---|---|---|---|---|---|
| [GrokSAR（DenoDet/V2 官方仓库）](https://github.com/GrokCV/GrokSAR) | **冲突**：README 写 CC BY-NC 4.0，侧栏和文件列表显示 Apache-2.0 | MMDet ≥3.0.0rc5,<3.1.0；mmcv <2.1；PyTorch 2.0.1；另需 mmrotate、mmyolo、mmpretrain、mmseg | 有 V1、V2（val）、V2（test）config；权重在百度和 OneDrive；每项有训练日志链接；[configs/gfl](https://github.com/GrokCV/GrokSAR/tree/master/configs/gfl) 和 26 种对比方法目录也在 | 115 star，33 commit，1 open / 39 closed | 见下 |
| [zcablii/SARDet_100K（MSFA）](https://github.com/zcablii/SARDet_100K) | CC BY-NC 4.0 | MMDet 3.1.0，mmcv 2.0.1，PyTorch 2.0.1 | 每行有 config 和百度权重；README 未链接完整日志；README 称目前只发布 train、val | 788 star，0 open / 62 closed，最近关闭 2026-01 | [#55](https://github.com/zcablii/SARDet_100K/issues?q=is%3Aissue+is%3Aclosed)、#54、#53 均为权重缺失 |
| [DEIM](https://github.com/ShihuaHuang95/DEIM)（本项目基座） | Apache-2.0 | 独立 PyTorch | 无 SAR 内容；支持自定义 COCO | 1.6k star，88 open | [#112](https://github.com/ShihuaHuang95/DEIM/issues/112) |
| [D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0（Objects365 权重另有条款） | 独立 PyTorch | 无；有 custom config | 3.4k star，154 open；README 最新日期 2024-11 | 未逐条读 |
| [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | **自定 DEIMv2 License**，商用需联系作者 | 独立 PyTorch，S 以上为 DINOv3 系骨干，需另下权重 | 无 | 2.1k star，36 open；最新 2026-10-22 | 未逐条读 |
| [RT-DETR/v2](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 | 独立 PyTorch | 无 | 5.7k star，411 open | 未逐条读 |
| [RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Apache-2.0 | 独立 PyTorch；训练需 DINOv3 ViT-B teacher 权重 | 无 | 679 star，19 open；2026-06 被 ECCV 2026 接收 | 未逐条读 |
| [RF-DETR](https://github.com/roboflow/rf-detr) | N/S/M/L 为 Apache-2.0；Atto/Femto/Pico/XL/2XL 为 PML 1.0 | pip 包 `rfdetr`，DINOv2 骨干；[总 batch 建议 16，分辨率须被 56 整除](https://rfdetr.roboflow.com/learn/train/) | 无 | 10k star，61 open | 未逐条读 |
| YOLO11/YOLO26 | [AGPL-3.0（据 RF-DETR 页面）](https://github.com/roboflow/rf-detr) | Ultralytics | 无官方配置 | — | — |
| [SARATR-X](https://github.com/waterdisappear/SARATR-X) | CC BY-NC 4.0 | MMDet（新版 mmcv 2.1.0） | 有 `configs/_hivit_/hivit_base_SARDet.py`，8 卡；README 未给 SARDet 数字；部分文件需邮件索取 | 277 star，2 open | — |
| [SARATR-X-v2](https://github.com/waterdisappear/SARATR-X-v2) | MIT | MMRotate 扩展，iTPN | 配置和日志在 `results/detection/`；**权重需邮件申请** | 15 star，2 open | — |
| [ViTP](https://github.com/zcablii/ViTP) | CC BY-NC 4.0 | MMRotate，mmcv-full 1.6.1 | 有 Cascade R-CNN config，权重在 ModelScope/HF | 179 star，2 open | README 写 57.9，[论文表写 59.7](https://arxiv.org/html/2509.17562v1.pdf) |
| [SARMAE](https://github.com/MiliLab/SARMAE)（CVPR 2026） | CC BY-NC 4.0 | mmrotate，PyTorch 1.12.1 | 页面只见预训练权重，未见 SARDet 微调权重 | 65 star，4 open | — |

- `GrokCV/DenoDet` 这个地址返回 404；[DenoDet V2 论文指向 GrokSAR](https://arxiv.org/html/2508.09392v1)。MSFA 仓库也放了一份 DenoDet 配置，路径为 [`MSFA/local_configs/DeonDet/gfl_r50_denodet_sardet.py`](https://github.com/zcablii/SARDet_100K/issues/50)，文件夹名的拼写就是这样。

**GrokSAR 配置和 issue 中的复现要点**

- DenoDet V2 val 配置：[GFL + FFTResNet-50，512×512，DAdaptAdam lr=1.0，WD 0.05，SyncBN，per-channel 均值约 36.5，std 约 52.1，`save_best='auto'`](https://raw.githubusercontent.com/GrokCV/GrokSAR/master/configs/DenoDetV2/DenoDetV2_1x_SARDet_100k.py)。
- V2 test 配置**不是同一套**：[1024×1024、partition_stride (16,16)、AdamW lr 1e-4](https://raw.githubusercontent.com/GrokCV/GrokSAR/master/configs/DenoDetV2/DenoDetV2_1x_SARDet_100k_test.py)；[维护者在 #37 确认了输入和 stride 的差异](https://github.com/GrokCV/GrokSAR/issues/37)。结果是 [val 56.71（52.47G），test 56.39（0.21T）](https://github.com/GrokCV/GrokSAR)。
- 数据集基础配置里 [`test_dataloader = val_dataloader`，训练集 `filter_empty_gt=True, min_size=16`，batch_size=16](https://raw.githubusercontent.com/GrokCV/GrokSAR/master/configs/_base_/datasets/SARDet_100k.py)。
- [#41（2026-06，open，无回复）](https://github.com/GrokCV/GrokSAR/issues/41)：有人用 D-FINE-S（640，b16，12 轮）对标论文，达不到论文 DETR 指标，512→640 也没有提升，并询问 132 轮是否算不公平比较。**这是找到的唯一一条 D-FINE 跑 SARDet-100K 的记录**，且没有给出数值。
- 其余：[#38 称 SAR-AIRcraft 复现高出约 16%，回复猜测是 val 而非 test](https://github.com/GrokCV/GrokSAR/issues/38)；[#40 指出 SAR-AIRcraft 的 mAP 其实是 AP50](https://github.com/GrokCV/GrokSAR/issues/40)。
- DAdaptAdam 来自 `dadaptation` 包，[3.0 版改动过 DAdaptAdam，可能与旧版结果不同](https://www.pypi.org/project/dadaptation/3.0/)。

**MSFA 配置与论文的差异**

- [论文 Table S12 写 AdamW、lr 1e-4、b16、12 轮，正文写 resize 800×800、用第 12 轮权重测 test](https://arxiv.org/html/2403.06534v3)。
- 但仓库 [`gfl_r50_IN_sup_sar.py` 写的是 SGD lr 0.01、`frozen_stages=1`、`norm_eval=True`，预训练路径是作者本机的绝对路径](https://raw.githubusercontent.com/zcablii/SARDet_100K/main/MSFA/local_configs/SARDet/other_detectors/gfl_r50_IN_sup_sar.py)。
- [#57](https://github.com/zcablii/SARDet_100K/issues/57)：`min_size=32` 的过滤让每轮只用 61,288 张，而不是 94,493 张。

**DEIM 复现**

- [#112](https://github.com/ShihuaHuang95/DEIM/issues/112)：X 模型 8×A100 跑到 56.16（论文 56.5）；作者称严格按官方配置时波动约 ±0.1；另一位用户称 54.7 难以稳定达到，54.4 更现实，并怀疑 AMP 的 NaN。本项目也为此做过 [FP32 重算处理](../../docs/sardet100k_full_baseline.md)。

### Inferences
- 只有 GrokSAR 能在一个仓库内用**同一套配置**同时复现 GFL 和 DenoDet V2，这是增量路线的真实优势。
- 但“训练集图数”在框架之间不一致：MMDet 配置会过滤，本项目用完整 94,493 张。这一点必须写进论文。
- MMDet 的 `batch_size` 按我的理解是每卡值，论文写的“16、四张 4090”无法确定全局 batch。要看 GrokSAR 日志的每轮迭代数才能定（未验证）。

### Gaps
- GrokSAR 的 LICENSE 文件正文没有读，许可证冲突未解决。
- 权重和日志托管在百度/OneDrive，按约束没有下载，也就没有验证日志内容。
- D-FINE、DEIMv2、RT-DETR、RF-DETR 的 issue 未逐条读；只确认 README 没有 SAR 结果。

## Q2 两张 RTX 4090 上的 SARDet-100K 训练成本

### Takeaway
一次实验的成本差约一个数量级：GFL/DenoDet V2 1x 约 **2–6 小时（估计）**，DEIM-D-FINE-M 102 轮约 **2–3 天（估计）**。6 组消融、单种子，前者不到 2 天，后者约 11–17 天。

### Cited Findings
- 固定协议：[DEIM-D-FINE-S，120 轮，全局 b32，640，2952 steps/epoch，共 354,240 steps；验证每轮 10,492 张](../../docs/sardet100k_full_baseline.md)。
- 官方 DEIM 轮数：[S 132（本项目定为 120）](../../docs/sardet_paper120_nozoom_nocrop.md)，本地配置中 `deim_hgnetv2_m_coco.yml` 为 102，`l`、`x` 为 58。
- 速度锚点一：本地 [DEIM-M、1024、b12，预计每轮 290 iter 约 2.6 分钟](../../docs/sar_protocol_b.md)，约 0.54 s/iter。这是该文档写的**预计值**。
- 速度锚点二：[MMDet 文档 Faster R-CNN 0.216 s/iter、RetinaNet 0.205 s/iter，硬件列为 8×V100](https://mmdetection.readthedocs.io/en/latest/model_zoo.html)，batch 未注明。
- 论文日程：[DenoDet V2 12 轮、b16、四张 4090](https://arxiv.org/html/2508.09392v1)；[MSFA 12 轮，Deformable DETR 50 轮、DETR 150 轮](https://arxiv.org/html/2403.06534v3)；[MDIS-DETR 150 轮、b16、AdamW 1e-3](https://www.mdpi.com/2072-4292/18/13/2076)；[SARLite 200 轮、b64、两张 4090](https://www.nature.com/articles/s41598-026-49143-5)；[Edge YOLO 300 轮、b16](https://www.mdpi.com/2072-4292/17/13/2168)；[SARES-DEIM 300 轮（其他 SAR 数据集）](https://arxiv.org/html/2604.04127)。
- [SARATR-X 预训练 800 轮、8 卡](https://github.com/waterdisappear/SARATR-X)；[v2 预训练 1200 轮、8 张 A800、总 batch 1600](https://arxiv.org/pdf/2607.23238)。

### Inferences

**成本估计表**（全部是估计。步数由图数和 batch 算出；s/iter 由上面两个锚点按像素和模型规模外推，误差可能到 ±50%；未含每轮验证，120 轮验证另加约 2–6 小时。M/L 的全局 batch 是我的假设，参照本项目 M 的 b28、800 输入 b20，24 GB 显存是否够 L b24 未验证）

| 方案 | 全局 batch | steps/epoch | 轮数 | 总步数 | s/iter（估） | 小时（估） |
|---|---|---|---|---|---|---|
| DEIM-D-FINE-S 640 | 32 | 2952 | 120 | 354,240 | 0.35–0.6 | 34–59 |
| DEIM-D-FINE-M 640 | 28 | 3374 | 102 | 344,148 | 0.45–0.7 | 43–67 |
| DEIM-D-FINE-M 800 | 20 | 4724 | 102 | 481,848 | 0.5–0.8 | 67–107 |
| DEIM-D-FINE-L 640 | 24 | 3937 | 58 | 228,346 | 0.55–0.85 | 35–54 |
| RT-DETR-R18 640（MDIS 日程） | 16 | 5905 | 150 | 885,750 | 0.12–0.2 | 30–50 |
| YOLO s 640（SARLite 日程） | 64 | 1476 | 200 | 295,200 | 0.15–0.3 | 12–25 |
| RF-DETR M（轮数未核实，暂按 50） | 16 | 5905 | 50 | 295,250 | 0.25–0.45 | 20–37 |
| GFL / DenoDet V2，512，1x | 16 或 32 | 5905 或 2952（过滤后更少） | 12 | 70,860 或 35,424 | 0.12–0.35 | 2–6 |
| DenoDet V2 test 配置，1024，1x | 同上 | 同上 | 12 | 同上 | 约 4 倍像素 | 10–20 |
| SARATR-X HiViT-B 微调，1x | 未知 | — | 12 | — | — | 8–15（把握低） |

- 预训练路线（800–1200 轮、8 卡以上）在两张 4090 上不现实，只能使用公开或索取的权重。
- DEIM-M 单种子 6 组消融约 11–17 天（估计）。可用现成的 [10% 子集筛选协议](../../docs/sardet_screening_handoff.md) 压缩，但子集排名能否代表全集，该文档自己也说没有证明。

### Gaps
- 本次**没有**读取服务器上的实际 `console.log`。全集 S 运行如果已经开始，其日志里的 `time:` 和 `eta:` 比这张表可靠得多，应直接替换。
- 项目文档里未找到 SARDet-100K 全集的墙钟时间或 AP。
- RF-DETR 推荐轮数没有查到。

## Q3 通用检测器在 SARDet-100K 上的排名；GFL 是否强于实时 DETR；DEIM-D-FINE 可能到多少

### Takeaway
**不存在同协议下 GFL 与实时 DETR 的直接比较。** 跨论文证据表明协议造成的差距比方法增益大：同一个 GFL 从 49.8 到 55.01，同一个 YOLOv10-S 从 65.1 到 68.84，同一个 RT-DETR-R18 从 54.24 到 57.1。**没有找到任何 DEIM 或 D-FINE 在 SARDet-100K 上的数字。** 我估计 DEIM-D-FINE-M 640 val AP 约 56–62，把握低，需要实跑确认。

### Cited Findings

**12 轮 MMDet 档**（[DenoDet V2 Table S7](https://arxiv.org/html/2508.09392v1)，512，val）：GFL 55.01、ATSS 54.95、TOOD 54.65、DDOD 54.02、AutoAssign 53.95、CenterNet 53.91、ConvNeXtV2 53.91、Cascade R-CNN 53.55、ConvNeXt 53.15、FCOS 52.52、Deformable DETR 52.00（50 或 12 轮未说明）、DETR 45.73、RetinaNet 46.48、Faster R-CNN 39.22、YOLOX 34.08；DenoDet V1 55.88，DenoDet V2 56.71。该表**不含 RT-DETR、D-FINE、DEIM**。

**同一方法的跨论文差异**

| 方法 | 结果 A | 结果 B | 条件差异 |
|---|---|---|---|
| GFL R50 | [49.8（MSFA，800，test）](https://arxiv.org/html/2403.06534v3) | [55.01（DenoDet V2，512，val，DAdaptAdam）](https://raw.githubusercontent.com/GrokCV/GrokSAR/master/configs/gfl/gfl_r50_fpn_1x_SARDet_100k.py) | 输入、优化器、BN、split 都不同；另有 [SM3Det 57.31（子集 + 联合训练，不可比）](https://arxiv.org/html/2412.20665) |
| Cascade R-CNN | [51.1（MSFA）](https://github.com/zcablii/SARDet_100K) | [53.55（V2 表）](https://arxiv.org/html/2508.09392v1) | 同上 |
| Faster R-CNN R50 | [49.0（MSFA）](https://github.com/zcablii/SARDet_100K) | [39.22（V2 表）](https://arxiv.org/html/2508.09392v1) | 差 10 个点，V2 表这一行明显偏低 |
| RT-DETR-R18 | [54.24 / AP50 85.89（MDIS-DETR，150 轮）](https://www.mdpi.com/2072-4292/18/13/2076) | [57.1 / 89.5（Edge YOLO，640，300 轮）](https://www.mdpi.com/2072-4292/17/13/2168) | 另有同文 RT-DETRv2-R18 **58.2 / 90.2** |
| YOLOv8 | [n 51.8（Edge YOLO）](https://www.mdpi.com/2072-4292/17/13/2168) | [s 63.7（SARLite）](https://www.nature.com/articles/s41598-026-49143-5/tables/1) | 规模不同 |
| YOLOv10 | [n 53.1 / 87.4（Edge YOLO）](https://www.mdpi.com/2072-4292/17/13/2168)、[s 65.1（SARLite）](https://www.nature.com/articles/s41598-026-49143-5/tables/1) | [N 59.47，S 68.84（SAR-NAS）](https://arxiv.org/html/2509.01279v1) | SAR-NAS 未公开代码和输入尺寸 |

**当前榜单上沿**（[SARATR-X-v2 Table X，GFL 微调，作者标注为投稿准备中](https://arxiv.org/pdf/2607.23238)）：MSFA ConvNeXt-B 56.4、SUMMIT 57.0、SARATR-X 57.2、SAMBA 59.7、ViTP 59.7、SARMAE 63.1、BabelRS 63.3、SARATR-X-v2 iTPN-B 63.4、iTPN-L **68.7**。注意 [SARMAE 自述用 Faster R-CNN 而非 GFL](https://arxiv.org/html/2512.16635)，表内检测头并不统一。另有 [MDIS-DETR 58.82（RT-DETR 系，150 轮）](https://www.mdpi.com/2072-4292/18/13/2076)，[SARLite 67.7（YOLO11 系，200 轮）](https://www.nature.com/articles/s41598-026-49143-5/tables/1)，[SAR-NAS-S 69.27](https://arxiv.org/html/2509.01279v1)。

**其他相关事实**
- 论文 [CGAQ-DETR 只训 7 轮](https://www.mdpi.com/2072-4292/17/18/3254)，所以它表里的 GFL 34.1、Deformable DETR 33.3、自身 38.6 不能当作性能参考。
- SARLite 的 [消融用 5 折交叉验证和配对 t 检验](https://www.nature.com/articles/s41598-026-49143-5)；而 [SARDet-100K 是 512 切片、重叠 200](https://arxiv.org/html/2403.06534v3)。
- 规模增益（其他 SAR 数据集）：[DEIM-S→DEIM-M，HRSID 73.0→75.6，SAR-Ship 71.3→71.6；同一表中 RT-DETR-L 为 59.1 和 67.3](https://arxiv.org/html/2604.04127)。COCO 上 [S 49.0、M 52.7、L 54.7、X 56.5](https://github.com/ShihuaHuang95/DEIM)。
- 项目 [10% 子集 S 最佳 AP 42.75（val 4096 张）](../../docs/sardet_offset_check.md)，训练只有 [9,449 张](../../docs/sardet_paper120_nozoom_nocrop.md)，**不能与全集结果比较**。

### Inferences
- MDIS-DETR 表只写“RT-DETR”（19.88M、56.97G），Edge YOLO 明写 RT-DETR-R18（20M、60G）。我按参数量把前者视为 R18，这是推断。
- GFL 55.01 不是“GFL 这个检测器的能力”，而是“GFL + 特定训练配方”的结果。DenoDet V2 的配方（512、数据集均值、SyncBN 解冻、DAdaptAdam）也许比其声称的模块增益（+1.7）更值钱，但这只是推断，没有隔离实验。
- 实时 DETR 是否强于 GFL：同为 R18 规模，RT-DETR 系的三个数字（54.24 / 57.1 / 58.2）与 GFL 55.01 上下交叉，**无法判定**。两者参数量也不同（约 20M 对 32M）。
- YOLO s 档的 64–69 明显高于 12 轮档的 55–57。可能的原因是长日程加 Mosaic，也可能是评估口径或数据划分不同（Ultralytics 评估未核实；随机折叠遇到重叠切片会泄漏）。在排除这些之前，不宜把它当作可达上限，也不宜视而不见。
- **DEIM-D-FINE 可达 AP 的估计**（val，把握低，全部是推断）：
  - S 640，120 轮：约 55–60（以 RT-DETR 系 54–58 为依据，COCO 上 S 比 R18 略高）。
  - M 640：约 56–62（在 S 上加 SARES-DEIM 观察到的 +0.3 至 +2.6）。
  - L 640：约 57–63，但官方只有 58 轮，日程短可能吃掉规模增益。
  - 800 输入：方向可能为正（[33% 目标属于 small](https://www.mdpi.com/2072-4292/17/18/3254)），但没有直接证据。DenoDet V2 的 1024 配置 test 56.39 与 512 配置 val 56.71 不同 split，不能说明问题；#41 反映 512→640 没有提升，但那是 12 轮。
- 结论是：这个基线**大概率进入 51–57 这一档，甚至超过**，但**很难单靠基线达到 63+ 档**，后者主要靠 SAR 域预训练。

### Gaps
- 没有 DEIM、D-FINE、DEIMv2、RT-DETRv4、RF-DETR 在 SARDet-100K 上的任何公开数字（搜索多次未找到）。
- 没有任何论文在同一协议下同时跑 GFL 和实时 DETR。MDIS-DETR 表中的非 DETR 行与 DenoDet V2 表逐字一致，未核实是否重训。
- Edge YOLO、SAR-NAS、SARLite 的评估工具（Ultralytics 或 pycocotools）和 split 未确认。

## Q4 审稿如何看待两条路线：已录用论文如何定位基线

### Takeaway
读到的 11 篇正文中，第三方方法**全部建立在通用检测器上**（GFL、Faster/Cascade R-CNN、RT-DETR、Deformable DETR、YOLO、DEIM）。在某个 SAR 方法上继续做的只有**原作者团队的续作**（DenoDet→V2、SARATR-X→v2）。这个样本有限，且 TCSVT、JSTARS 正文没有读到。

### Cited Findings

| 论文 | 状态 | 基线 | 如何说明基线 | 比较方式 |
|---|---|---|---|---|
| [DenoDet V2](https://arxiv.org/html/2508.09392v1) | AAAI 2026 | GFL，插件模块 | 称“generic object detection network”，**未解释为何选 GFL** | 25 种方法同框架 12 轮；有无模块的消融 |
| [SARMAE](https://arxiv.org/html/2512.16635) | CVPR 2026 | Faster R-CNN（预训练贡献） | 直接沿用 SARATR-X 的微调协议 | 表中混有文献方法 |
| [RSAR](https://arxiv.org/html/2501.04440v1) | CVPR 2025 | H2RBox-v2（FCOS 系） | 对比方法“use the default configuration in MMRotate” | 插件式增量，并发布新数据集 |
| [O²-DEIM/O²-DFINE](https://arxiv.org/html/2603.15497) | DOI 指向 TGRS 2026；光学遥感 | D-FINE、DEIM | 没有明文理由 | **同骨干配对**，所有模型 72 轮、b8；逐组件消融 |
| [SARATR-X-v2](https://arxiv.org/pdf/2607.23238) | 投稿准备中 | GFL | **明确论证**：两阶段 RPN 对 speckle 虚警敏感，GFL 无此环节 | 表中多为文献结果 |
| [SM3Det](https://arxiv.org/html/2412.20665v2) | 页面未写 venue | GFL | “superior performance on horizontal SAR object detection” | 与各数据集单模型比较 |
| [MDIS-DETR](https://www.mdpi.com/2072-4292/18/13/2076) | Remote Sensing 2026 | RT-DETR | 称 RT-DETR 在复杂 SAR 中“has not been systematically studied” | 消融以 RT-DETR 为基准；自身 150 轮，对比表与 12 轮 DenoDet V2 行并列 |
| [CGAQ-DETR](https://www.mdpi.com/2072-4292/17/18/3254) | Remote Sensing 2025 | Deformable DETR | 针对 SAR 目标数量和尺度波动 | 声称参数一致；7 轮 |
| [SARLite](https://www.nature.com/articles/s41598-026-49143-5) | Sci. Reports 2026 | YOLO11s | 主打轻量 | 5 折 CV + t 检验 |
| [Edge YOLO](https://www.mdpi.com/2072-4292/17/13/2168) | Remote Sensing 2025 | YOLOv8n | 主打边缘部署 | 含 RT-DETR 对比 |
| [SARES-DEIM](https://arxiv.org/html/2604.04127)（预印本） | arXiv 2026 | DEIM-S | “direct architectural baseline” | **明确说明跨家族结果只是参考，不是受控比较**；SAR-Ship 上只比 DEIM-S 高 0.4 |

补充（只读到摘要或片段，未读正文）：[MD-DETR，TGRS 2024，RT-DETR 系，三级去噪](https://unpaywall.org/10.1109%2FTGRS.2024.3489212)，在 [MDIS 表中为 55.38](https://www.mdpi.com/2072-4292/18/13/2076)。

### Inferences
- 审稿层面的重点似乎不是“基线是不是 SAR 专用”，而是**基线是否同协议、消融是否干净、能否说明 SAR 机理**。两类顶会论文都没有详细论证基线，SARATR-X-v2 那样的机理论证是可借鉴的写法。
- 第三方在 DenoDet V2 上做增量的已发表样例**没有找到**。这种做法的风险是：论文的新颖性依赖一个 +1.7 的模块，且对比上限由原作者的配置决定。这是推断，不是审稿人的陈述。
- 实时 DETR 路线的**可写空位**：MDIS-DETR 明确把 RT-DETR 写成未充分研究；D-FINE/DEIM 在 SARDet-100K 上没有公开基准。但这个空位可能很快被占，[SARES-DEIM 已在其他 SAR 数据集上这样做](https://arxiv.org/html/2604.04127)。
- 12 轮与 120 轮的日程差异是预期会被质疑的点（[#41 已有人公开提出](https://github.com/GrokCV/GrokSAR/issues/41)）。O²-DEIM 的“所有模型同轮数”是最稳妥的模板，但对 MMDet 方法跑 120 轮有额外成本（约 20–50 小时，估计）。
- 文献中常见直接把旧表数字搬过来（MDIS、SAR-NAS 的数值与 V2 表一致）。这在 MDPI 能通过，不代表 TGRS、CVPR 审稿也接受。

### Gaps
- **TCSVT、JSTARS 的相关正文一篇也没有读到**；TGRS 只读到光学遥感的 O²-DEIM。IEEE Xplore 无法访问，且搜索没有返回 SAR + SARDet-100K 的 TGRS 正文。
- 没有审稿意见的一手材料，以上是由已发表论文反推。
- CVPR 2024 SFS-Conv 全文返回 403，未读。

## Q5 把 DenoDet 类模块移植进 DEIM 的难度，以及与 MMDetection 论文的公平比较

### Takeaway
**移植难度低**：核心模块是纯 PyTorch，mm 依赖只在 ResNet 外壳上；估计 1–2 天完成代码，难点在 AMP 和插入位置，而不是框架。**同一 `val.json` 上比较 COCO 指标是可行的**，但只有 val 档可直接比；不少文献数字来自 test。

### Cited Findings

**上游代码**（[`FFTresnet.py`](https://raw.githubusercontent.com/GrokCV/GrokSAR/master/groksar/models/backbones/FFTresnet.py)）
- `FFTTransformerattentionlayer`、`SpatialPatchCrossTransformerattentionlayer`、`CrossTransformerAttention` 都是 `nn.Module`，**不依赖 mm**。
- mm 依赖（`BaseModule`、`build_conv_layer`、`build_norm_layer`、`ResLayer`、`MODELS.register_module`）只出现在 `BasicBlock`、`Bottleneck`、`FFTResNet`。
- 调用位置：stem 后一次，每个 stage 后一次，共 5 个实例；**不在残差块内部**。所有实例使用相同的 `partition_stride`。
- 该参数实际是**分块数**（默认 8×8），注意力维度等于分块数（64），故参数量随分块数平方增长，与通道数无关。
- `img_scale` 已被注释掉、未使用；输入尺寸不固定，无法整除时先插值再还原。
- 上游**没有 autocast 或 dtype 处理**。
- 前向：`fft2 → fftshift → 幅度/相位 → sin/cos → 两次交叉注意力门控 → 单位化 → 重组 → ifft2 → real → ReLU`。

**本地集成点**
- 注册：[`engine/core/workspace.py`](../../engine/core/workspace.py) 的 `@register`，YAML 组合。
- 插入：[`engine/deim/hybrid_encoder.py`](../../engine/deim/hybrid_encoder.py) `forward` 中已有 `rem`、`pacfg`、`pafp`、`pgd_rem` 等开关，模式是 `use_xxx=False` 参数，在 `proj_feats[...]` 上就地调用（第 667–678 行）。
- 先例：[`engine/deim/phase_aligned_frequency.py`](../../engine/deim/phase_aligned_frequency.py) 共 90 行，已用 `torch.fft`，其注释说明**整个模块在 AMP 下以 FP32 运行再还原 dtype**。
- 骨干备选：[`engine/backbone/hgnetv2.py`](../../engine/backbone/hgnetv2.py)，554 行。
- 本地相关信号：项目已[决定不再跟进 CLCR-Lite、W-FDR、PACFG](../../docs/sar_dfine_paper_plan_2026-11-03.md)（来自 SAR-AIRcraft 任务）。

**评估**
- 本地：[`coco_eval.py`](../../engine/data/dataset/coco_eval.py) 使用 `faster_coco_eval`；后处理 [`num_top_queries: 300`](../../configs/base/dfine_hgnetv2.yml)。
- GrokSAR：[`CocoMetric`，`Annotations/val.json`，bbox](https://raw.githubusercontent.com/GrokCV/GrokSAR/master/configs/_base_/datasets/SARDet_100k.py)；[`score_thr=0.05`、NMS 0.6、`max_per_img=100`](https://raw.githubusercontent.com/GrokCV/GrokSAR/master/configs/gfl/gfl_r50_fpn_1x_SARDet_100k.py)。
- val 集一致性：本地 [val 10,492 张、22,703 个目标，train 198,747 个目标](../../docs/sardet100k_full_baseline.md)，与 [CGAQ-DETR Table 1、3 的统计一致](https://www.mdpi.com/2072-4292/17/18/3254)。
- 噪声参考：本地[同一模型日志 AP 42.7546 与导出重算约 42.88，差约 0.125](../../docs/sardet_offset_check.md)。

**文献数字属于哪个 split**
- val：[DenoDet V2 56.71、GFL 55.01](https://arxiv.org/html/2508.09392v1)；[MAE ViTDet 54.9（明确 val，512）](https://arxiv.org/html/2501.11249v1)。
- test：[MSFA 系列（第 12 轮权重）](https://arxiv.org/html/2403.06534v3)；[DenoDet V2 test 56.39](https://github.com/GrokCV/GrokSAR)。
- [ViTP 表中 DenoDetv2 写 56.4](https://arxiv.org/html/2509.17562v1.pdf)，即 test 值，与 MSFA 的 test 值一致；而 MDIS-DETR 表用的是 val 档数字（GFL 55.01）。[README 又称只发布了 train、val](https://github.com/zcablii/SARDet_100K)，但 [GrokSAR 有 test 配置](https://github.com/GrokCV/GrokSAR/tree/master/configs/_base_/datasets)。

### Inferences
- **工作量（估计）**：
  - 抽出 3 个纯 PyTorch 类并加 `use_xxx` 开关：约半天。
  - 处理 FP32、核对初始化（`-t` 只加载匹配键，新增模块应为新参数）、写配置：约半天到 1 天。
  - 这与 PACFG 的既有模式相同。
- **需要决策的语义差异**：上游放在 ResNet 各 stage 后（含 stem），放进 HGNetv2 才算忠实移植；放 `proj_feats` 上只是“受启发”，论文里要如实写。S5 在 640 下是 20×20，8×8 分块无法整除，会走插值路径。
- **在 DETR 上的增益不能假定**：该模块的 +1.7 是在 GFL 上测得的；本地频域模块（PACFG）已有不再跟进的记录。
- 移植代码前先确认 GrokSAR 许可证。如果按 CC BY-NC，最稳妥的是像 PACFG 那样独立实现并注明参考。
- **公平比较的最低要求**（我的建议）：
  1. 官方 GrokSAR 配置在 MMDet 环境自跑 GFL 和 DenoDet V2，写同一 `val.json`。
  2. 把两边预测导出为 COCO JSON，用**同一个**评估脚本重算。差异约 0.1 量级可以接受，并报告。
  3. 表格明确分列 val 和 test，不混用文献数字；文献行标明“原文报告”。
  4. 权重选择规则预先固定：DenoDet 的 `save_best='auto'` 与 `test=val` 都在同一集合上选最优，DEIM 的 best_stg 也是。两边同样处理并明说。若使用 test，只评一次。
  5. 明确写出输入尺寸、训练图数（过滤与否）、轮数。可加一行“MMDet GFL 延长到同等轮数”回应日程质疑。

### Gaps
- 没有实测 `faster_coco_eval` 与 pycocotools 在此数据上的差异。
- 本地是否有 `test.json` 未确认（文档只提及 train、val）。
- GrokSAR 自定义数据集类 `SAR_Det_Finegrained_Dataset` 的代码未读，可能影响类别映射。
- 未实际尝试移植，工作量仅为估计。

## Q6 决策表与建议

### Takeaway
**建议：继续以 DEIM-D-FINE 作为主基线（通用检测器路线），主力规模选 M；同时把 GFL 和 DenoDet V2 作为 MMDet 校准行自跑，不在其上做增量。** 这是有条件的建议：以全集 baseline 的实测 AP 为前提，反转条件见下。

### Cited Findings
决策表依据的事实均已在 Q1–Q5 给出来源，此处不重复。

### Inferences

**决策表**（评分是我的判断；小时为估计）

| 选项 | 单次成本 | 代码/复现风险 | 预期基线 AP（val） | 论文空位 | 主要风险 | 判断 |
|---|---|---|---|---|---|---|
| **DEIM-D-FINE-M**（本项目） | 43–67 h | 低：Apache，基础设施已就绪；风险在 AMP 数值 | 56–62（低把握） | **高**：未见公开基准 | 消融慢；基线可能不够强；日程差异受质疑 | **首选** |
| DEIM-D-FINE-S | 34–59 h | 同上 | 55–60（低把握） | 高 | 比 M 低；已在跑（协议已定） | 作为 M 之前的闸门和轻量行 |
| DEIM-D-FINE-L | 35–54 h | 同上 | 57–63（低把握） | 高 | 58 轮偏短 | 最终方法的规模对照行 |
| D-FINE（plain） | 与 DEIM 相近 | 低；README 已久未更新 | 未知 | 高 | — | 只做一行对照，[既有计划已包含](../../docs/sar_dfine_paper_plan_2026-11-03.md) |
| DEIMv2 | 高于同规模（估计） | 中：自定许可证，DINOv3 权重 | 未知，SAR 上无证据 | 中 | 换基座后本地模块需重验 | 一行对照即可 |
| RT-DETRv2/v4 | 30–50 h（R18 估） | v2 低；v4 需 teacher 权重 | v2-R18 54–58（已有文献） | 低（MDIS、MD-DETR 已占） | 不新 | 对照行 |
| RF-DETR | 20–37 h（估） | 中：框架完全不同；只 N/S/M/L 为 Apache | 未知 | 中 | 本地模块无法复用 | 不选 |
| GFL（MMDet） | 2–6 h | 中：环境锁定；55.01 的配方依赖 DAdaptAdam | 55.01（val，文献） | 低：已饱和 | 换框架；增益空间小 | **校准行**，不做主线 |
| Cascade R-CNN | 约 5–10 h（估） | 同上 | 51–53.6（文献） | 低 | 慢且不强 | 不选 |
| YOLO s | 12–25 h | 中：AGPL；文献数字互相矛盾 | 64–69（口径存疑） | 低：文章多、期刊层次参差 | 口径和泄漏风险 | 需要先校准，不选主线 |
| **DenoDet V2 增量** | 3–6 h | 中：许可证冲突，val/test 两套配置，#41 无回复 | 56.71（文献） | 低：+1.7 模块上再加增量 | 上限由原作者配方决定，未见第三方先例 | **校准行**，不做主线 |
| SARATR-X / v2 增量 | 8–15 h（把握低） | 高：权重需索取，预训练 8 卡不可复现 | 57–68.7（文献） | 中 | 重大算力依赖；不可独立复现 | 不选；**但可作为初始化实验的探索** |

**具体步骤建议**
1. 先拿到全集 DEIM-D-FINE-S 120 轮的实际 AP 和 s/iter，用它替换本文所有估计。
2. 同步在独立 MMDet 环境自跑 `gfl_r50_fpn_1x_SARDet_100k.py` 和 `DenoDetV2_1x_SARDet_100k.py`（各约 2–6 小时，估计）。这能一次回答两个问题：55.01/56.71 能否复现，以及两个框架的评估是否一致。
3. S 达标后跑 M。消融只在 M 上做，子集筛选后全集确认。
4. 论文定位：“SARDet-100K 上首个 D-FINE/DEIM 系实时 DETR 基准 + SAR 机理改进”。报告表分三组：同协议自跑（DEIM、D-FINE、GFL、DenoDet V2）；官方 12 轮协议；文献数字（标明 split，不参与排名）。正视 SARMAE、SARATR-X-v2 等 63+ 的预训练结果，把它们写成另一类路线（重型预训练、ViT），不回避。

**什么证据会反转这个建议**（阈值是我的判断，不是文献标准）

| 证据 | 转向 |
|---|---|
| DEIM-D-FINE-M 640 全集 val AP 明显低于 **约 54**（低于 MDIS 的 RT-DETR-R18 54.24） | 基线不可用；先排查，再考虑转 GFL 路线 |
| 同一 val.json、同一评估下，自跑 GFL 高出 DEIM-M **≥2 AP**，且 M 800 也追不上 | 基线故事不成立，转 GFL + 模块 |
| 单次 M 实测超过约 **5 天**，或可用 GPU 少于 2 张，或论文截止不足约 2 个月 | 成本不允许，转 MMDet 路线（可 2–6 小时一轮） |
| 目标期刊或审稿明确要求 12 轮同协议比较，且 DEIM 在 12 轮下明显落后 | 转 GFL 系 |
| 发现已发表或预印本的 DEIM/D-FINE on SARDet-100K 且高于本方案 | 空位消失，定位要改 |
| 自跑 GFL 无法复现 55.01（差 >1 AP） | 进一步削弱增量路线，**加强**本建议，并成为论文可写的发现 |
| 能取得 SARATR-X-v2 权重且作为 DEIM 初始化 AP 大幅提升 | 考虑以“SAR 预训练 + 实时 DETR”为主线（骨干适配成本高） |
| 确认 YOLO s 64–69 在官方 split、标准 COCO 评估下可复现 | DEIM 档位显得偏低，须重新评估目标检测器 |

### Gaps
- 决策的最大不确定性是 DEIM 全集实测 AP，目前没有；建议本身依赖这一点。
- 审稿证据没有覆盖 TCSVT、JSTARS（见 Q4）。
- 只有 MMDet 系和 Ultralytics 系的跨论文间接比较，没有同协议结果。
