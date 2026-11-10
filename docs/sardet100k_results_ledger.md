# SARDet-100K 实验总表

这是 SARDet-100K 的唯一结果表。每个实验结束后在这里加一行，协议和启动命令见
[协议文档](sardet100k_m_protocol.md)。生成的权重、日志和评估文件留在服务器，不提交 Git。

## 固定协议

除“方法”和表中注明的种子外，以下设置在所有 DEIM-D-FINE-M 实验中保持不变：

| 项目 | 设置 |
| --- | --- |
| 数据 | 官方划分：train 94,493 / val 10,492 / test 11,613，六类 |
| 模型 / 初始化 | DEIM-D-FINE-M（HGNetv2-B2），`-t` 载入 COCO M 权重 |
| 训练 | 36 轮，全局 batch 32（每卡 16），GPU 0、1 |
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
| E002 | GREA 保护环证据注意力 | 0 | | | | | | | | | | | | | | | | | | 筛选中 |
| E000-old | 基线（旧服务器） | 0 | 35 | 65.15 | — | 93.22 | 69.97 | | | | | | | | | | | 19.48M | | 基线（旧服务器） |
| E003 | Q-MAL 像素容差质量目标（旧服务器，τ=1.5） | 0 | 35 | 64.51 | −0.64 | 94.04 | 69.27 | | | | | | | | | | | 19.48M | | 淘汰（对照 E000-old） |
| E007 | 去掉 MAL 与 Dense O2O（VFL，无 Mosaic/MixUp；旧服务器） | 0 | 35 | 66.01 | +0.86 | 93.70 | 70.68 | | | | | | | | | | | 19.48M | | 完成（对照 E000-old；诊断性对照，不是方法） |

六类 AP 来自独立重算（总 AP 65.29，与训练日志一致）。

## test

| ID | 方法 | 种子 | 权重 | AP | AP50 | AP75 | APs | APm | APl | AR100 | ship | aircraft | car | tank | bridge | harbor |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E000 | 基线 | 0 | best_stg2 | 62.39 | 90.92 | 66.35 | 59.01 | 74.65 | 71.72 | 70.88 | 72.90 | 49.80 | 71.90 | 53.91 | 44.87 | 80.95 |
| E000-old | 基线（旧服务器） | 0 | best_stg2 | 62.7 | 91.6 | 66.1 | 58.9 | 75.0 | 72.4 | 71.2 | | | | | | |
| E003 | Q-MAL（旧服务器，τ=1.5） | 0 | best_stg2 | 61.4 | 91.5 | 65.3 | 57.4 | 73.0 | 71.2 | 69.6 | | | | | | |
| E007 | 去掉 MAL 与 Dense O2O（旧服务器） | 0 | best_stg2 | 63.1 | 90.9 | 66.6 | 61.0 | 75.1 | 69.3 | 71.8 | | | | | | |

## 运行记录

| ID | 配置 | Git revision | 运行目录（新服务器 `/root/zxy/experiments/`） |
| --- | --- | --- | --- |
| E000 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml` | `455b196` | `deim_m_sardet100k_deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_seed0_20261105_125147_N2zAJh` |
| E001 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_m1_geometry_36e_b32.yml` | `8e83207` | `deim_m_sardet100k_deim_hgnetv2_m_sardet100k_m1_geometry_36e_b32_seed0_20261106_145049_1Gh5VI` |
| E002 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_grea_36e_b32.yml` | `10015d1` | `deim_m_sardet100k_deim_hgnetv2_m_sardet100k_grea_36e_b32_seed0_20261106_161421_p4o6KB` |
| E000-old | 旧服务器上补跑的 E000 协议（见运行目录 `resolved_config.yml`） | 未记录 | 旧服务器 `/data2/zxy/sar/experiments/deim_m_sardet100k_deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_seed0_20261107_083221_byTPNu` |
| E003 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_qmal_36e_b32_old_server.yml` | `07e932b` | 旧服务器 GPU 0、1，`/data2/zxy/sar/experiments/deim_m_sardet100k_*qmal*`，已完成 |
| E007 | `configs/deim_dfine/deim_hgnetv2_m_sardet100k_no_mal_no_dense_o2o_36e_b32_old_server.yml` | 见启动目录 `start_command.sh` | 旧服务器 GPU 0、1，`/data2/zxy/sar/experiments/deim_m_sardet100k_*no_mal_no_dense_o2o*`，已完成 |

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

### E000-old 旧服务器基线（2026-11-08）

stg1 最佳为第 31 轮 val AP 63.88，stg2 最佳为第 35 轮 65.15，按实测 val AP 选 `best_stg2` 评估 test。test 只有 COCO 汇总输出（三位小数），六类 AP 尚未独立重算。与新服务器 E000 相比：val −0.14，test +0.3，均在单种子波动内。E003（Q-MAL）在旧服务器运行，只和 E000-old 作差。

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
