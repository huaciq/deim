# SARDet-100K 实验总表

这是 SARDet-100K 的唯一结果表。每个实验结束后在这里加一行，协议和启动命令见
[协议文档](sardet100k_m_protocol.md)。生成的权重、日志和评估文件留在服务器，不提交 Git。

## 固定协议

除“方法”和表中注明的种子外，以下设置在所有 DEIM-D-FINE-M 实验中保持不变：

| 项目 | 设置 |
| --- | --- |
| 数据 | 官方划分：train 94,493 / val 10,492 / test 11,613，六类 |
| 模型 / 初始化 | DEIM-D-FINE-M（HGNetv2-B2），`-t` 载入 COCO M 权重 |
| 训练 | 36 轮，全局 batch 32（每卡 16），双卡；具体 GPU 编号见运行记录 |
| 输入 | 基础和验证 640；训练多尺度 480..800，第 32 轮起固定 640 |
| 优化 | AdamW，主 LR 4e-4，骨干 4e-5，WD 1e-4；flatcosine，warmup 2000 步，flat 20 轮 |
| 增强 | Mosaic、MixUp 各 p=0.5（第 4–20 轮），PhotometricDistort p=0.5（第 4–32 轮），翻转 p=0.5；关闭 ZoomOut、IoUCrop |
| EMA / AMP | EMA 0.9999；FP16，FP32 loss，非有限批次一次 FP32 重算 |
| 评估 | COCO bbox，每轮完整 val；按 val AP 选权重 |

## 记录规则

- **val：** 训练日志中 AP 最高的一轮，轮号从 0 开始。
- **test：** 只评估最终方法和基线，不用于筛选模块或调参。其他实验的 test 列留空。
- **Δ：** 只和同协议、同种子的基线作差。有多个基线种子后，增加均值和波动。
- **状态：** 基线 / 筛选中 / 晋级 / 淘汰 / 最终。
- **晋级门槛：** 暂定 36 轮 val AP 涨 0.5 以上，基线有第二个种子后按波动调整。

## 主表：val

| ID | 方法 | 种子 | 最佳轮 | AP | ΔAP | AP50 | AP75 | APs | APm | APl | AR100 | ship | aircraft | car | tank | bridge | harbor | 参数量 | 训练时长 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E000 | 基线 | 0 | 35 | 65.29 | — | 93.15 | 70.26 | 60.18 | 76.63 | 72.09 | 73.36 | 73.75 | 62.78 | 71.21 | 57.66 | 45.97 | 80.40 | 19.48M | 17:38 | 基线 |
| E001 | M1 成像—标注几何解耦 | 0 | | | | | | | | | | | | | | | | | | 中止（第 2 轮后，用户决定不作为方法，无结论） |
| E002 | GREA 保护环证据注意力 | 0 | 35 | 64.37 | −0.92 | 93.28 | 69.03 | 59.20 | 75.76 | 69.71 | 72.42 | 73.44 | 61.48 | 71.23 | 57.47 | 45.85 | 76.73 | 19.92M | 18:48 | 淘汰 |
| E000-old | 基线（旧服务器） | 0 | 35 | 65.15 | — | 93.22 | 69.97 | | | | | | | | | | | 19.48M | | 基线（旧服务器） |
| E003 | Q-MAL 像素容差质量目标（旧服务器，τ=1.5） | 0 | 35 | 64.51 | −0.64 | 94.04 | 69.27 | | | | | | | | | | | 19.48M | | 淘汰（对照 E000-old） |
| E007 | 去掉 MAL 与 Dense O2O（VFL，无 Mosaic/MixUp；旧服务器） | 0 | 35 | 66.01 | +0.86 | 93.70 | 70.68 | | | | | | | | | | | 19.48M | | 完成（对照 E000-old；诊断性对照，不是方法） |
| E004 | 匹配分类代价 2→0.5（E1；系数 1 的运行已中止） | 0 | 35 | 65.28 | −0.01 | 87.99 | 73.20 | 58.93 | 77.36 | 71.77 | 78.28 | 74.28 | 64.96 | 72.07 | 56.72 | 44.35 | 79.32 | 19.48M | 18:13 | 淘汰 |
| E005 | 框监督改为层内匹配（关闭并集） | 0 | | | | | | | | | | | | | | | | | | 暂停，未运行 |
| E006 | E004 + E005 | 0 | | | | | | | | | | | | | | | | | | 暂停，未运行 |
| E008 | 去掉 Dense O2O（MAL 保留） | 0 | 35 | 65.86 | +0.57 | 93.19 | 70.24 | 61.63 | 77.44 | 71.13 | 72.98 | 74.63 | 61.18 | 71.68 | 59.23 | 48.64 | 79.80 | 19.48M | 18:01:28 | 诊断性对照 |
| E009 | RegionDense 保尺度区域拼图 | 0 | 35 | 66.12 | +0.83 | 93.62 | 70.76 | 62.18 | 77.74 | 70.62 | 73.00 | 75.39 | 62.23 | 71.51 | 59.09 | 48.73 | 79.77 | 19.48M | 17:54:10 | 未晋级（相对 E008 val +0.26） |
| E010 | RegionDenseSAR v2 背景相容与源图遍历 | 0 | 35 | 66.45 | +1.16 | 93.72 | 71.08 | 62.89 | 78.16 | 71.37 | 73.91 | 75.37 | 62.77 | 70.44 | 59.46 | 49.13 | 81.55 | 19.48M | 17:33:40 | 晋级（对 E008 val +0.59；辅助候选） |
| E011 | RegionDense + MixUp（E009 上只把 MixUp 概率 0→0.5） | 0 | | | | | | | | | | | | | | | | 19.48M | | 暂缓，未运行（见 [val 拆解](sardet100k_region_dense_val_breakdown.md)） |
| E012 | 只加方向增强（垂直翻转、90° 旋转），结构不变 | 0 | | | | | | | | | | | | | | | | | 19.48M | | 暂缓，未运行（D18 原始裁块结果不作为完整检测器结论） |
| E013 | 方向共享编码与方向证据融合（v1 历史实现，原图 + 垂直翻转） | 0 | | | | | | | | | | | | | | | 19.58M | | 已实现，未运行；保留用于历史复现，正式实验转入 E013-v2 |
| E013-v2 | 方向共享骨干与投影空间证据融合（原图 + 垂直翻转，单次 AIFI/FPN/PAN） | 0 | | | | | | | | | | | | | | | 19.58M | | 训练中（8 卡服务器 GPU 4、5，2026-11-12 21:19 启动；对照 E000） |

六类 AP 来自独立重算（总 AP 65.29，与训练日志一致）。

## test

| ID | 方法 | 种子 | 权重 | AP | AP50 | AP75 | APs | APm | APl | AR100 | ship | aircraft | car | tank | bridge | harbor |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E000 | 基线 | 0 | best_stg2 | 62.39 | 90.92 | 66.35 | 59.01 | 74.65 | 71.72 | 70.88 | 72.90 | 49.80 | 71.90 | 53.91 | 44.87 | 80.95 |
| E000-old | 基线（旧服务器） | 0 | best_stg2 | 62.7 | 91.6 | 66.1 | 58.9 | 75.0 | 72.4 | 71.2 | | | | | | |
| E003 | Q-MAL（旧服务器，τ=1.5） | 0 | best_stg2 | 61.4 | 91.5 | 65.3 | 57.4 | 73.0 | 71.2 | 69.6 | | | | | | |
| E007 | 去掉 MAL 与 Dense O2O（旧服务器） | 0 | best_stg2 | 63.1 | 90.9 | 66.6 | 61.0 | 75.1 | 69.3 | 71.8 | | | | | | |
| E002 | GREA（已淘汰） | 0 | best_stg2 | 61.75 | 90.71 | 66.07 | 58.38 | 73.85 | 69.19 | 70.08 | 72.44 | 50.21 | 71.59 | 53.94 | 44.49 | 77.83 |
| E004 | 匹配分类代价 0.5（已淘汰） | 0 | best_stg2 | 61.8 | 85.1 | 69.0 | 58.2 | 74.8 | 70.1 | 75.7 | | | | | | |
| E008 | 去掉 Dense O2O | 0 | best_stg2 | 63.32 | 91.10 | 67.11 | 60.97 | 75.25 | 71.49 | 70.72 | 73.41 | 49.61 | 71.79 | 56.61 | 46.98 | 81.50 |
| E009 | RegionDense | 0 | best_stg2 | 63.43 | 90.96 | 67.57 | 61.94 | 75.17 | 68.39 | 70.92 | 73.80 | 49.16 | 72.13 | 56.29 | 47.31 | 81.87 |
| E010 | RegionDenseSAR v2 | 0 | best_stg2 | 63.88 | 91.65 | 67.22 | 62.07 | 75.89 | 69.60 | 71.83 | 73.56 | 50.55 | 72.14 | 56.97 | 47.51 | 82.53 |

E008、E009、E010 的 test 依用户明确要求评估；未用 test 选择 checkpoint 或调整配置。

E002 的 test 是淘汰决定之后按用户要求评估的，没有参与筛选。E004 的 test 是训练结束后按用户要求评估的，同样没有参与筛选；只有 COCO 汇总输出（三位小数）。

## 运行记录

| ID | 配置 | Git revision | 运行目录（新服务器 `/root/zxy/experiments/`） |
| --- | --- | --- | --- |
| E000 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml` | `455b196` | `deim_m_sardet100k_deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_seed0_20261105_125147_N2zAJh` |
| E001 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_m1_geometry_36e_b32.yml` | `8e83207` | `deim_m_sardet100k_deim_hgnetv2_m_sardet100k_m1_geometry_36e_b32_seed0_20261106_145049_1Gh5VI` |
| E002 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_grea_36e_b32.yml` | `10015d1` | `deim_m_sardet100k_deim_hgnetv2_m_sardet100k_grea_36e_b32_seed0_20261106_161421_p4o6KB` |
| E004 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_match_cls05_36e_b32.yml` | `51c280b` | 8 卡服务器 GPU 4、5：`deim_m_sardet100k_deim_hgnetv2_m_sardet100k_match_cls05_36e_b32_seed0_20261108_195504_7ewHSg`（之前的两次启动已中止，见[设计文档](sardet100k_matching_ablation.md)） |
| E000-old | 旧服务器上补跑的 E000 协议（见运行目录 `resolved_config.yml`） | 未记录 | 旧服务器 `/data2/zxy/sar/experiments/deim_m_sardet100k_deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_seed0_20261107_083221_byTPNu` |
| E003 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_qmal_36e_b32_old_server.yml` | `07e932b` | 旧服务器 GPU 0、1，`/data2/zxy/sar/experiments/deim_m_sardet100k_*qmal*`，已完成 |
| E007 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_no_mal_no_dense_o2o_36e_b32_old_server.yml` | 见启动目录 `start_command.sh` | 旧服务器 GPU 0、1，`/data2/zxy/sar/experiments/deim_m_sardet100k_*no_mal_no_dense_o2o*`，已完成 |
| E004（系数 1，已中止） | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_match_cls1_36e_b32.yml` | `34ff56d` | 8 卡服务器：`deim_m_sardet100k_deim_hgnetv2_m_sardet100k_match_cls1_36e_b32_seed0_20261108_183430_XLvsRU`（第 1 轮内停止，无结果） |
| E005 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_layer_box_36e_b32.yml` | `34ff56d` | 暂停，未运行 |
| E006 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_match_cls1_layer_box_36e_b32.yml` | `34ff56d` | 暂停，未运行 |
| E008 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_no_dense_o2o_36e_b32.yml` | `7c3e899` | `deim_m_sar_aircraft_deim_hgnetv2_m_sardet100k_no_dense_o2o_36e_b32_seed0_20261109_180613_bVK0K3` |
| E009 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_36e_b32.yml` | `6e305c3` | `regiondense_deim_hgnetv2_m_sardet100k_region_dense_36e_b32_seed0_20261110_203618_BjgqFF` |
| E010 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32.yml` | `33172bb` | `regiondense_sar_v2_deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32_seed0_20261111_214950_thYezk`（训练与 test 已完成，GPU 4、5） |
| E011 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_mixup_36e_b32.yml` | 待提交 | 暂缓，未运行 |
| E013-v2 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_orientation_fusion_v2_36e_b32.yml` | `5e90eb1` | `orientfusion_v2_deim_hgnetv2_m_sardet100k_orientation_fusion_v2_36e_b32_seed0_20261112_211929_zPjIqO`（GPU 4、5，训练中） |
| E013 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_orientation_fusion_36e_b32.yml` | `c59bd6e` | v1 历史配置，未运行 |

## 备注

### E000 指标核对（2026-11-06）

- 用独立推理和 AP 代码重算：val 65.29，test 62.34（评估流程 62.39）。
- 训练全程没有报错，也没有触发 FP32 重算。
- 诊断用的目标裁片相关性（≥0.9）：test 50.4% 的目标在训练集有高相似裁片，这组 AP 68.51，其余 42.95；val 分别为 57.2%、70.67、44.48。
  按整图相似度（≥0.95）分组，test 差距小得多：66.01 对 63.60。两组的难度可能本来就不同，所以这个差距不能全部归因于重复图。
  只作诊断，官方划分不变。结果在运行目录的 `independent_verify_20261106/report.json`。
- test aircraft 49.80，明显低于 val 的 62.78，原因未查。

### E000 错误诊断

见 [基线诊断](sardet100k_baseline_diagnosis.md)。

### E002 判读（2026-11-07）

未达到晋级门槛，按预定标准淘汰，不补种子，也不做消融。判读见 [GREA 设计](sardet100k_grea_design.md#结果与判定e002)。六类 AP 来自独立重算（总 AP 64.37，与训练日志一致），结果在运行目录的 `independent_verify_20261106/`。

### E000-old 旧服务器基线（2026-11-08）

stg1 最佳为第 31 轮 val AP 63.88，stg2 最佳为第 35 轮 65.15，按实测 val AP 选 `best_stg2` 评估 test。test 只有 COCO 汇总输出（三位小数），六类 AP 尚未独立重算。与新服务器 E000 相比：val −0.14，test +0.3，均在单种子波动内。E003（Q-MAL）在旧服务器运行，只和 E000-old 作差。

### E004 匹配分类代价 0.5（2026-11-09）

- **选权重：** stg1 最佳为第 31 轮，stg2 最佳为第 35 轮 val AP 65.28，按 val 选 `best_stg2`。
- **相对 E000 的 val：** AP −0.01，AP50 −5.16，AP75 +2.94，AR100 +4.92。
- **相对 E000 的 test：** AP −0.6（61.8），AP50 −5.8，AP75 +2.65，AR100 +4.8。
- **结论：** 未达到 +0.5 的晋级门槛；预定的机制指标 2、3 也没有改善，淘汰。
- **原因：** 匹配器确实选中了更好的框（OGSOD tank 末层匹配 IoU .645 → .698），但一对一去重失效：分数 ≥ .3 的重复框从 387 个增加到 5,991 个，OGSOD tank 从 79 个增加到 1,532 个。
- **训练时长：** 18:13 含约 9% 的匹配统计开销。完整报告见[设计文档](sardet100k_matching_ablation.md#结果)。

### E003 Q-MAL（2026-11-09）

stg1 最佳第 31 轮 val AP 63.14，stg2 最佳第 35 轮 64.51 / 94.04 / 69.27，按 val 选 `best_stg2`。对 E000-old：val AP −0.64、AP50 +0.82、AP75 −0.70；
test AP −1.3（61.4）、AP50 −0.1、AP75 −0.8、APs −1.5、APm −2.0、APl −1.2。未达 +0.5 晋级门槛，且下降超过约 0.4 的跨服务器波动，淘汰。
判读：τ=1.5 让 ≥16 px 且误差 ≤1–2 px 的框目标分饱和到 ≈1，分数不再区分“准”和“很准”，高 IoU 阈值下的类内排序变差（AP50 略升、AP75 降）；
小目标也没有受益（test APs 降）。说明 MAL 用严格 IoU 作目标带来的定位排序价值大于它对小目标分数的压低，原假设不成立。
机制指标（分数差距）未测，因 AP 已下降，不影响结论。不再扫描 τ（属调参）。

### E007 去掉 MAL 与 Dense O2O（2026-11-09）

stg1 最佳第 31 轮 val AP 65.64，stg2 最佳第 35 轮 66.01 / 93.70 / 70.68，按 val 选 `best_stg2`。对 E000-old：val AP +0.86、AP50 +0.48、AP75 +0.71；
test AP +0.4（63.1）、AP50 −0.7、AP75 +0.5、APs +2.1（61.0）、APm +0.1、APl −3.1（69.3）、AR100 +0.6。单种子。
判读：在 SARDet-100K 的 36 轮协议下，DEIM 的 MAL 与 Dense O2O 合起来没有带来增益，val 反而低 0.86；小目标受损（test APs −2.1），大目标受益（APl +3.1）。
与 Mosaic 诊断（把目标缩到像素极限、跨来源拼接）方向一致，但这里两个机制同时去掉，不能单独归因。E008（只去 Dense O2O，8 卡服务器）对照 E000 后可分开；
MAL 的单独作用只能跨服务器推算（约 0.4 AP 波动）。

### E008 去掉 Dense O2O（2026-11-10）

8 卡服务器 GPU 4、5，`7c3e899`。stg1 最佳第 31 轮 val AP 65.42，stg2 最佳第 35 轮 65.86 / 93.19 / 70.24，按 val 选 `best_stg2`。
对同服务器的 E000：val AP +0.57、AP50 +0.04、AP75 −0.02、APs +1.45、APm +0.81、APl −0.96、AR100 −0.38；
test AP +0.93（63.32）、AP50 +0.18、AP75 +0.76、APs +1.96（60.97）、APm +0.60、APl −0.23。单种子。
按类（val，见主表）：bridge +2.67、tank +1.57、ship +0.88、car +0.47、harbor −0.60、aircraft −1.60。
判读：只关掉 Mosaic 与 MixUp、保留 MAL，val 就高 0.57，增益在小目标和中目标，按类看在 bridge 和 tank，aircraft 下降。E007（同时去掉 MAL）在旧服务器上是 +0.86，
两者相差 0.29，在约 0.4 的跨服务器波动之内，所以 E007 的增益主要来自去掉 Dense O2O，MAL 单独的作用仍然无法从这两次运行中分开。

### E008 与 E009 RegionDense（2026-11-11）

同新服务器、同 36 轮/b32/seed0 协议，均按第 35 轮（从 0 开始）的最高 val AP 选 best_stg2；test 完整 11,613 张，使用相同验证流程。
E008 val 65.86 / test 63.32；E009 val 66.12 / test 63.43。E009 相对 E008：val +0.26、test +0.11、test AP75 +0.46、APs +0.96、APl -3.10；tank -0.32、bridge +0.33。
相对 E000：E009 val +0.83、test +1.04、APs +2.93，但不能用较弱 E000 替代 E008 这个直接对照。
未达到预设 E008 +0.5 的 val 晋级线，当前记录为未晋级。单种子局部收益不能作为稳定整体提升的结论。
训练时长 E000 17:37:45、E008 18:01:28、E009 17:54:10；首次达到 val AP 65 的轮号（从 1 开始）分别为 35、30、28。
详细指标、各类 AP、原始评估目录与复现命令见 [RegionDense 结果](sardet100k_region_dense_results.md)。

### E010 RegionDenseSAR v2 结果（2026-11-12 记录）

在 E009 保尺度、保邻域基础上，引入保持源图完整遍历的 FIFO 组图和背景相容选择。新服务器 GPU 4、5，36 轮/global batch 32/seed0，其余训练及评估协议与 E008/E009 相同。训练用时 17:33:40；按第 36 轮（index 35）的最高 val AP 66.45 选择 best_stg2 的 EMA 权重，完成完整 test 11,613 张图评估。

test AP 63.88，较 E008 +0.56，较 E009 +0.45；APs 62.07，较 E008 +1.09；APl 69.60，较 E009 +1.21，但仍比 E008 低 1.89。相对 E008 六类 test AP 均上涨。达到 E008 +0.5 val AP 筛选线，记录为晋级的辅助候选；当前为单种子，两项改动的独立贡献尚未分开。
分类指标、实际评估目录和重跑命令见 [RegionDenseSAR 评估结果](sardet100k_region_dense_sar_results.md)；方法、CPU 验证、训练/恢复命令见 [正式实验](sardet100k_region_dense_sar_experiment.md)。
