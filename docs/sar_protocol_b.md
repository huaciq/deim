# SAR-AIRcraft 主协议 B-v1：trainval 训练、统一尺度 1024、固定日程

日期：2026-11-03。依据：[协议 v2 与 A/B 方案](sar_protocol_v2.md)、[文献协议调研](sar_aircraft_protocol_survey.md)、
[型号混淆线索诊断](sar_confusion_cues_diagnosis.md)。旧协议（640 冻结、方案 A）结果全部保留，作为诊断与对照。

## 1. 协议

| 项目 | 设定 | 依据 |
|---|---|---|
| 训练集合 | 官方 trainval（`COCO/trainval.json`，3489 张，13513 个目标） | 官方数据集论文"训练集 13513 / 测试集 2950"、SAR-Net"默认划分" |
| 报告 | 官方 test（879 张）AP50 为主，另报 AP、AP75、各类 AP50；val 已在训练集内，不作为指标 | 与官方、SAR-Net 同口径；test 有 77% 目标在 trainval 中出现，属数据集本身性质，论文中说明 |
| 选权重 | 不选：`fixed_schedule: True`，取最后一轮 `final.pth` | val 在训练集内；640 运行中 final 与 val 选权重在 test 上一致 |
| 输入 | 每张图右/下补边到 1500×1500 后缩放到 1024：所有图同一缩放比 0.683；测试相同 | 数据集 1 m 分辨率，同类飞机在各尺寸源图中像素尺寸一致；直接缩放会使缩放比相差最多 1.9 倍 |
| 多尺度 | 保留 DEIM 批级多尺度 768–1280（同一 batch 内所有图同一尺度），第 120 轮停止 | 既往单种子对比 64.5 对 58.7（有/无多尺度）；批级缩放不破坏图间尺度一致 |
| 增强 | 只保留水平翻转 + DEIM mixup（第 4–70 轮）；关闭 Mosaic、PhotometricDistort、ZoomOut、IoUCrop | Mosaic 对每张图单独缩放到固定尺寸并随机 0.5–1.5 缩放，破坏统一尺度；SAR 单通道，色调/饱和度无效，对比度抖动改变斑点统计；ZoomOut/IoUCrop 改尺度且黑边不像杂波 |
| 初始化 | COCO M 权重（`-t`），7 个类别头全部由 COCO airplane 一行初始化 | 原 80→7 映射失败，类别头随机；单卡前 3 轮 AP 0.1/2.3/3.4 → 4.8/7.7/10.6 |
| 日程 | 132 轮：warmup 600 iter（约 2 轮，二次曲线），flat 到第 70 轮，cosine 到第 120 轮，最后 12 轮无增强 | 原 2000 iter 二次 warmup 约 18 轮、半程时 LR 仅峰值 26%，前 15 轮几乎不学；1024 下 102 轮未收敛 |
| 优化 | AdamW，主 LR 2.3e-4、骨干 2.3e-5，WD 1e-4，全局 batch 12（每卡 6），EMA 0.9999 | 1024 实测每卡 6 张峰值 13.4 GB；EMA 后期稳定高 1.3–1.5 AP（640 checkpoint 实测） |
| 精度 | AMP FP16 + FP32 loss/retry | 沿用 |
| 模型 | DEIM-D-FINE-M，seed 0 | 沿用 |

配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_b_canvas1024_132e_b12_old_server.yml`。
实现：`PadToCanvas`（`engine/data/transforms/_transforms.py`，补边后把 `orig_size` 设为画布尺寸，预测框按画布映射回原图像素；
左上对齐，原图坐标不变）；`tools/init_head_from_coco_class.py`（生成类别头初始化权重）；solver 的 `fixed_schedule` 与 `final.pth`。

**已知局限与预期**

- 统一尺度让 A330/A320 等尺寸差异大的类别更容易区分；但诊断显示主要混淆对（737/787、A220/ARJ21）的源图像素尺寸本身重叠
  （可分性 0.51–0.64），这一项对它们的帮助预计有限。
- 800 图补边后有效区域只占画布 28%，计算有浪费；这是统一尺度的代价。
- 一次同时改了多项设定，作为"协议"整体与旧配方比较；单项贡献如需写入论文，再在此协议上做去掉单项的消融。
- 预计每轮约 2.6 分钟（290 iter），132 轮约 5.5–6 小时；显存与已测 1024 运行相同（张量尺寸一致）。

## 1.1 B-v1 baseline 结果（seed 0）

运行：`/home/zxy/sar/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_b_canvas1024_132e_b12_old_server_seed0_20261103_193451_xwONF9`，
`final.pth`（epoch 131）在官方 test 上评估，输出在该目录的 `test_eval_final/`。

| AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---:|---:|---:|---:|---:|---:|---:|
| 68.09 | 87.91 | 74.64 | 61.4 | 67.6 | 67.3 | 81.4 |

| 类别 | AP | AP50 | AP75 |
|---|---:|---:|---:|
| A220 | 65.79 | 87.45 | 66.71 |
| A320/321 | 81.28 | 100.00 | 95.38 |
| A330 | 84.50 | 95.95 | 95.95 |
| ARJ21 | 60.22 | 81.08 | 63.89 |
| Boeing737 | 61.39 | 87.62 | 61.98 |
| Boeing787 | 62.22 | 80.01 | 73.22 |
| other | 61.23 | 83.27 | 65.32 |

按预定判定（test AP50 ≥ 约 83）冻结 B-v1：之后的方法和消融都在这组配置上比较，不再按 test 调配方。

判读：

- 与只用 train 的旧配方（640，AP 45.74 / AP50 65.37）相比 AP +22.35、AP50 +22.54，但同时改了训练集合和配方，
  两者各占多少未单独测量（"旧配方 + trainval"对照决定不跑）。预计训练集合是主要来源：test 中 train+val 见过的目标占 77%（只用 train 时 50%），
  737 从 23% 升到 72%，其 AP 从约 23 升到 61.39。
- 第 131 轮 train_loss 与 val AP 仍缓慢变化（val 在训练集内，只反映拟合），不能据此判断 test 是否收敛；不用 test 选轮数，日程固定为 132 轮。
- 单种子。test 去重后约 381 架飞机，且 77% 目标在训练集中有近似副本，模块比较需要多种子，并按见过/未见过拆开看。

## 1.2 B-v1 错误拆解（seed 0，test）

`tools/diagnose_detection_errors.py`（§3.2），`final.pth` float32 导出，AP50 87.93（§1.1 为 87.91）。
"见过"指 trainval 中存在相关 ≥0.9 的近似切片。oracle 列是假设修好对应错误后的 AP50，只用于比较上限。

| 子集 | GT | AP50 | 去掉型号混淆 | 去掉背景误检 | 两者都去掉 | 类别无关 AP50 | 定位率 score≥0.3 / ≥0.05 | 定位后型号正确率 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 全部 | 2950 | 87.93 | 91.47 | 90.71 | 95.09 | 88.95 | 0.869 / 0.900 | 0.961 |
| 见过 | 2282 | 95.73 | 96.66 | 98.21 | 99.27 | 94.94 | 0.982 / 0.993 | 0.989 |
| 没见过 | 668 | 28.85 | 45.51 | 38.49 | 72.35 | 43.13 | 0.484 / 0.584 | 0.771 |

| 类别 | GT / 没见过 | AP50 | 去掉型号混淆 | 见过 / 没见过 AP50 | 没见过定位率 | 没见过型号正确率 | 没见过漏检（约） |
|---|---:|---:|---:|---|---:|---:|---:|
| A220 | 460 / 103 | 87.07 | 92.86 | 96.40 / 32.49 | 0.786 | 0.395 | 22 |
| A320/321 | 52 / 0 | 100.00 | 100.00 | 100.00 / — | — | — | 0 |
| A330 | 31 / 0 | 95.95 | 96.41 | 95.95 / — | — | — | 0 |
| ARJ21 | 362 / 92 | 81.18 | 83.20 | 95.97 / 12.70 | 0.207 | 1.000 | 73 |
| Boeing737 | 550 / 152 | 87.77 | 90.12 | 94.46 / 51.06 | 0.704 | 0.785 | 45 |
| Boeing787 | 454 / 113 | 80.04 | 89.94 | 97.63 / 23.27 | 0.257 | 0.931 | 84 |
| other | 1041 / 208 | 83.47 | 87.74 | 89.69 / 24.73 | 0.418 | 1.000 | 121 |

判读：

- 损失集中在没见过的目标：score≥0.3 的 386 个漏检中约 345 个（89%）属于这 23% 的目标。
- 这些目标主要是检不出，其次才是型号判错；类别无关 AP50 也只有 43.13，因此 **Q4 放弃**。
- **ARJ21 不是型号混淆问题**：定位后型号正确率 1.000，去掉型号混淆只到 83.20，没见过的只定位到 0.207。此前按细粒度混淆解释 ARJ21 差距的说法不成立。
- 主要型号混淆只有两对：A220→other 50（42 个没见过）、737→787 22（21 个没见过）。
- 背景误检与记忆无关：见过子集去掉后也从 95.73 升到 98.21。
- 局限：单种子。这些结果来自 test，只用于判断机制方向，不用于调参或选配方。

**漏检原因（`tools/diagnose_missed_targets.py`）：**

| 没见过的 668 个目标 | 已定位 | 有框但分数低 | 框不准 | 没有候选 |
|---|---:|---:|---:|---:|
| 数量 | 323 | 294 | 27 | 24 |

- **不是 query 选择或回归问题**：85% 的漏检已有 IoU≥0.5 的框（框 IoU 中位数 0.71），只是分数低。分数不是差一点：中位数 0.009，而已定位的是 0.784，所以调阈值或校准没有用。
- **"没见过"基本名副其实**：640 个在训练集没有相关 ≥0.8 的匹配；相关 0.8–0.9 的 28 个全部定位。截断（切片边缘 0.42 对 0.50）、尺寸、源图尺寸、场景（全部没见过的图 0.535，反而高于混合图 0.449）都解释不了。对比度有影响但不决定（低 / 中 / 高 0.43 / 0.38 / 0.64）。
- **按架次是全有或全无**：test 共 381 架不同的飞机，没见过的 82 架。≥3 个切片的 61 架里，GT 数上"全部检出 224、全部漏检 233、部分检出 176"。部分检出的架次中，检到和没检到的切片在位置、边缘、对比度上没有明显差别。
- **有效样本极少**：没见过的 ARJ21 只有 10 架、787 只有 14 架。一架 19 个切片的 ARJ21 全部漏检，就占该类 GT 的 5%。追上 SAR-Net 需要的 2.29，粗算相当于多检出三四架飞机，所以单种子差值的波动可能很大。
- **朝向有关联但不决定**：按同类 trainval 中相近朝向的比例分三档，没见过目标的定位率 0.40 / 0.47 / 0.68（主轴）、0.38 / 0.43 / 0.62（对称轴）。分类别不一致：other 0.19→0.69，ARJ21 0.11→0.26，737 和 787 没有这个趋势或相反。切片不独立，这只是相关。
- **背景误检有一部分像漏标**：264 个（score≥0.3），other 占 158。用户目视最高分的 32 个框（2026-11-04）：A220 一组 4 个最像漏标飞机，能辨出机身、机头、两翼和尾部；亮团一组 4 个较像；暗目标一组 9 个有嫌疑；其余存疑或是多目标大框。这些框是少数物体的重复切片，不是独立飞机。去掉背景误检的 +2.78 因此不能全当作可达到的空间。官方标注不改，只在分析中说明。

## 1.3 斑点重采样 A（方法，不是协议的一部分）

`SpeckleResample`（`engine/data/transforms/_transforms.py`）在翻转之后、补边之前作用于存储的像素，概率 0.5，第 4–119 轮生效，最后 12 轮看原图。
配置 `deim_hgnetv2_m_sar_aircraft_b_canvas1024_speckle_132e_b12_{old,new}_server.yml` 相对 B-v1 只多这一项。

- **模型**：存储值的分位数符合幅度域 Nakagami，不符合对数正态（下表），等效单视数约 1.3。按每张图实测的 (均值/标准差)² 取形状参数，空间相关由高斯模糊控制。
- **结构保留**：Lee 权重保留强散射、替换均匀杂波。标准权重把小而亮的目标压低约 25%，所以乘以 structure_gain=2。
  饱和亮块内部方差小，会被当成杂波而变成空心环（真实图拼图上发现，统计量看不出来），所以局部均值超过全图中位数 3–6 倍时逐渐完全保留（bright_ratio=3）。
- **参数**：只用 trainval 的 150 张图（`tools/check_speckle_resampling.py`，新服务器 `diagnostics/speckle_check_bright_20261104_163103`），训练前定下，不看 test。

| 版本 | 原图 ENL | 工作分辨率 ENL | lag1 | lag2 | q0.01 | q0.5 | q0.99 | 杂波残差相关 | 目标框残差相关 | 目标裁剪与原图 ≥0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原图 | 4.67 | 7.08 | 0.550 | 0.062 | 0.226 | 0.949 | 2.057 | — | — | — |
| **gain 2、窗口 7、σ 0.8、bright 3（采用）** | 4.15 | 5.86 | 0.583 | 0.051 | 0.209 | 0.944 | 2.075 | 0.201 | 0.385 | 0.944 |

未保留亮块的对照（`speckle_check_g*_20261104_160646`）：gain 1 杂波残差相关 0.096、目标框 0.205，但亮目标被压低；窗口 15 的 ENL 最接近（4.74 / 7.01），目标框残差相关却升到 0.606，单张耗时约 3 倍。

局限：

- 生成图的 ENL 比原图低 11–17%，噪声略强。单张约 0.28 秒（服务器负载下、最低优先级），按概率 0.5、每卡 4 个 worker 估算不应成为瓶颈，启动后看 `data:` 时间确认。
- **A 只检验"模型记住斑点"这一种记忆。** 斑点层确实换掉了（杂波残差相关 0.201），但目标的强散射布局保留，94% 的目标裁剪与原图相关仍 ≥0.9。如果模型记住的是散射布局，A 不会有效，这是一个可以被证伪的结果。
- 判定：没见过子集的定位率和 AP50 要上升；只是见过子集下降则不采用。单种子 AP50 增益 <1 不算有效。

## 1.4 B-v1 + REM 结果（seed 0，新服务器）

运行：`/root/zxy/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_b_canvas1024_rem_132e_b12_new_server_seed0_20261104_102922_JpWE66`，
revision `aef3c03`，torch 2.1.2、torchvision 0.16.2，`final.pth` 在 test 上评估；诊断在 `diagnostics/errors_rem_20261104_165217`。

| 指标 | B-v1 | REM | 差值 |
|---|---:|---:|---:|
| AP | 68.09 | 67.5 | −0.6 |
| AP50 | 87.91 | 86.90 | −1.01 |
| AP75 | 74.64 | 74.8 | +0.2 |

| 类别 AP50 | A220 | A320/321 | A330 | ARJ21 | Boeing737 | Boeing787 | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| B-v1 | 87.45 | 100.00 | 95.95 | 81.08 | 87.62 | 80.01 | 83.27 |
| REM | 86.71 | 99.35 | 95.52 | 80.40 | 87.19 | 73.37 | 85.77 |
| 差值 | −0.74 | −0.65 | −0.43 | −0.68 | −0.43 | **−6.64** | +2.50 |

| 诊断（float32 导出，AP50 87.93 对 87.02） | B-v1 | REM |
|---|---:|---:|
| 见过 AP50 | 95.73 | 94.24 |
| 没见过 AP50 | 28.85 | 26.45 |
| 类别无关 AP50（全部 / 没见过） | 88.95 / 43.13 | 90.05 / 47.64 |
| 定位率 score≥0.3（全部 / 没见过） | 0.869 / 0.484 | 0.881 / 0.519 |
| 定位后型号正确率（全部 / 没见过） | 0.961 / 0.771 | 0.951 / 0.706 |
| 737→787（全部 / 没见过） | 22 / 21 | 42 / 39 |

判读：

- 按采用规则不采用。旧协议下的 +1.80 没有在 B-v1 上复现。
- 787 的见过 AP50 从 97.63 掉到 89.40，几乎解释了全部下降。原先写的"REM 多检出了目标、737→787 翻倍"经新服务器基线复核（§1.8）基本是运行间波动，已撤回。
- 单种子、两台服务器（B-v1 在旧服务器）。用户此前在两台服务器上跑过相同实验，波动不大，因此决定按失败处理，不再跟进 REM（2026-11-04）。

## 1.5 90° 旋转 R（方法，不是协议的一部分）

`RandomRot90`（`engine/data/transforms/_transforms.py`）在翻转之后、补边之前作用，概率 0.5，从 90°、180°、270° 中均匀选一个，第 4–119 轮生效，最后 12 轮保持原朝向。
配置 `deim_hgnetv2_m_sar_aircraft_b_canvas1024_rot90_132e_b12_{old,new}_server.yml` 相对 B-v1 只多这一项。参数在训练前定下，之后不按 test 调。

- **动机**：trainval 的停放朝向集中（§1.2、调研文档 §7.2），没见过的飞机朝向越少见越容易漏检。不加新数据时，这是最直接的扩充朝向的办法。
- **为什么只用 90° 的倍数**：框保持紧贴、尺度不变，不破坏统一画布。任意角度只有水平框标注时会把框撑大。
- **覆盖范围**：与现有水平翻转组合覆盖全部 8 种朝向。竖直翻转 = 水平翻转 + 旋转 180°，所以不单独加。

局限：

- 旋转 90° 和 270° 会交换距离向与方位向，飞机相对雷达的方位和阴影方向也随之改变，物理上不严格。如果 R 有效，再跑 `angles: [180]` 的消融，它只含翻转组合，不交换两个方向。
- 朝向与漏检的关联只是相关，分类别也不一致（other 强、ARJ21 弱、737 相反），R 不一定能解决 ARJ21。
- 判定与 A 相同：没见过子集的定位率和 AP50 要上升；只是见过子集下降则不采用；单种子 AP50 增益 <1 不算有效。另外按朝向常见程度分档看，少见档应改善最多。

## 1.6 分数证据与误检分组（REM 权重，零训练）

`tools/diagnose_score_evidence.py`，新服务器 `diagnostics/evidence_rem_20261104_194319`。用的是 REM 的 `final.pth`，B-v1 的权重在旧服务器，未做。

**"定位确定性重打分"不成立，放弃。** 低分的没见过目标，边框分布并不比重复框更确定，比误检更不确定：

| 查询组 | 数量 | score 中位数 | concentration 中位数 | entropy 中位数 |
|---|---:|---:|---:|---:|
| 见过且已定位 | 2251 | 0.840 | 0.229 | 0.410 |
| 没见过且已定位 | 347 | 0.778 | 0.207 | 0.480 |
| 没见过且低分 | 304 | 0.011 | 0.177 | 0.610 |
| 重复框 | 15254 | 0.0007 | 0.188 | 0.560 |
| 背景 | 237215 | 0.0002 | 0.109 | 0.801 |
| 误检 | 286 | 0.637 | **0.247** | **0.321** |

- concentration 的 AUC：低分目标对背景 0.632，对重复框 0.391，对误检 0.069。
- 按 logit(score) + w·z(证据) 重打分，所有证据、所有权重的 AP50 都低于基准（85.38 → 85.19 到 78.72），没见过子集也下降。这是对 test 取的上限，上限都是负的，所以不用实现。
- 这也削弱了 CLCR-Lite 在 B-v1 上的理由（输入是同一组统计量），不再单独安排运行。
- **B-v1 权重复核（旧服务器，用户运行）**：结论相同。concentration 的 AUC 为对背景 0.665、对重复框 0.469、对误检 0.043；所有重打分都低于基准（AP50 86.70 → 86.53 到 79.54）。

**"强散射硬负样本"证据不足，暂不实现。** 286 个误检来自 75 个物体。对比度中位数 17.1 dB，只比 GT 的 15.1 dB 略高。目视拼图（只经本人判读）：多数不是廊桥、建筑那样的强散射，而是暗的类飞机图案、成列的亮点，以及框住多架飞机的大框；长条亮结构只占少数。把它们当负样本训练，有压低真实飞机的风险。

B-v1 权重下 264 个误检来自 74 个物体，对比度中位数 17.6 dB（GT 15.1 dB）。**两个独立训练的模型误检的基本是同一批物体**：按尺寸和对比度近似匹配，B-v1 的前 20 个物体全部能在 REM 中找到对应，全部 74 个中有 55 个（把对比度平移 3 dB 的对照只有 16 个）。这是近似匹配，没有逐框核对，但说明这些误检不是随机噪声，而是稳定存在的结构。其中还有一些是贴在切片边缘的被截断目标（目视）。

**疑似漏标影响很小。** 用户目视判断 G10、G19、G22、G43、G59 像漏标飞机（2026-11-04），共 19 个框（占 7%）。把它们移出误检后 AP50 从 86.70 升到 86.98，只多 0.28（只含移出误检的影响，没有把这些目标算作新增的 GT）。其余 69 个物体、245 个框没有被判为飞机，所以漏标不是差距的主要来源，论文里只作小项说明。官方标注和官方指标不变。

硬负样本仍不实现：匈牙利匹配已经把训练图上所有未匹配的高分查询当负样本监督，所以挖掘难例本来就在发生。test 上的误检大多是另一批没见过的结构，和没见过的飞机属于同一个泛化问题。

## 1.7 B-v1 + 斑点重采样 A 结果（seed 0，新服务器）

运行：`/root/zxy/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_b_canvas1024_speckle_132e_b12_new_server_seed0_20261104_165916_Vu7o7t`，
revision `b8a0468`，`final.pth` 在 test 上评估；诊断在 `diagnostics/errors_speckle_20261104_234809`。

| 指标 | B-v1 | A | 差值 |
|---|---:|---:|---:|
| AP | 68.09 | 67.8 | −0.3 |
| AP50 | 87.91 | 87.44 | −0.47 |
| AP75 | 74.64 | 74.8 | +0.2 |

| 类别 AP50 | A220 | A320/321 | A330 | ARJ21 | Boeing737 | Boeing787 | other |
|---|---:|---:|---:|---:|---:|---:|---:|
| B-v1 | 87.45 | 100.00 | 95.95 | 81.08 | 87.62 | 80.01 | 83.27 |
| A | 85.63 | 99.89 | 96.72 | 80.63 | 86.42 | 77.42 | 85.40 |
| 差值 | −1.82 | −0.11 | +0.77 | −0.45 | −1.20 | −2.59 | +2.13 |

| 诊断（float32 导出，AP50 87.93 对 87.53） | B-v1 | A |
|---|---:|---:|
| 见过 AP50 | 95.73 | 95.21 |
| 没见过 AP50 | 28.85 | 26.51 |
| 没见过类别无关 AP50 | 43.13 | 40.83 |
| 没见过定位率 score≥0.3 | 0.484 | 0.467 |
| 定位后型号正确率（全部 / 没见过） | 0.961 / 0.771 | 0.960 / 0.760 |

判读：

- 按 §1.3 的判定不采用：AP50 没有涨，没见过子集的定位率和 AP50 也都没有上升。
- 模型的泛化问题不靠换斑点解决，"记住斑点"不是主要机制。Q1 斑点一致性依赖同一假设，不再实现；频域抗噪类方向也失去这条依据。
- 模型仍可能记住强散射布局，A 没有改变它，所以这不否定实例记忆本身。
- 单种子。REM 和 A 都是 787 下降、other 上升，新服务器基线也有这个方向（各约 1 个点），见 §1.8。

## 1.8 新服务器 B-v1 基线（seed 0）与同机比较

运行：`/root/zxy/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_b_canvas1024_132e_b12_new_server_seed0_20261104_235935_gDNGNC`，
revision `75b38ba`，`final.pth` 在 test 上评估；诊断在 `diagnostics/errors_bv1_new_20261105_083216`。配置与旧服务器只差路径。

| 指标 | B-v1 旧服务器 | B-v1 新服务器 | REM（新） | A（新） |
|---|---:|---:|---:|---:|
| AP | 68.09 | 67.7 | 67.5 | 67.8 |
| AP50 | 87.91 | 87.55 | 86.90 | 87.44 |
| AP75 | 74.64 | 74.5 | 74.8 | 74.8 |
| A220 AP50 | 87.45 | 86.67 | 86.71 | 85.63 |
| A320/321 AP50 | 100.00 | 100.00 | 99.35 | 99.89 |
| A330 AP50 | 95.95 | 95.06 | 95.52 | 96.72 |
| ARJ21 AP50 | 81.08 | 80.35 | 80.40 | 80.63 |
| Boeing737 AP50 | 87.62 | 87.51 | 87.19 | 86.42 |
| Boeing787 AP50 | 80.01 | 79.01 | 73.37 | 77.42 |
| other AP50 | 83.27 | 84.26 | 85.77 | 85.40 |
| 见过 AP50（导出） | 95.73 | 94.84 | 94.24 | 95.21 |
| 没见过 AP50（导出） | 28.85 | 27.29 | 26.45 | 26.51 |
| 类别无关 AP50（导出） | 88.95 | 89.52 | 90.05 | 88.68 |
| 没见过定位率 score≥0.3 | 0.484 | 0.534 | 0.519 | 0.467 |
| 没见过 ARJ21 定位率 | 0.207 | 0.402 | 0.380 | 0.228 |
| 没见过定位后型号正确率 | 0.771 | 0.745 | 0.706 | 0.760 |
| 737→787 | 22 | 36 | 42 | 26 |

判读：

- 同配方同种子，总体 AP50 只差 0.36，与用户此前的经验一致；但这一对运行分不开服务器差异和运行随机性。
- 同机比较：REM AP50 −0.65，A −0.11。两者仍都没有达到采用标准，结论不变。A 的差值在波动范围内，准确说法是"没有效果"，不是"变差"。REM 的 787 同机仍低 5.64，是唯一明显超出这一对波动的单类变化。
- **子集和单类指标的波动远大于总体**：没见过定位率 0.484 对 0.534，没见过 ARJ21 定位率 0.207 对 0.402，737→787 22 对 36，都出自同一配方。这与"按架次全有或全无"一致。因此 §1.2 的具体数字（如 ARJ21 0.207）只代表那一次运行，定性结论（损失集中在没见过的目标）在两次运行中一致。
- 之后判断方法时，单类差 ≤1、没见过定位率差 ≤0.05、没见过 ARJ21 这类小子集的变化，都不能单独作为依据（只来自一对运行的粗略尺度，不是种子方差）。

## 2. 运行顺序

基线命名为 DEIM-D-FINE-M（D-FINE 架构 + DEIM 的 MAL、MixUp 与 COCO 权重），不因后续结果更改。

1. B-v1 baseline（本配置，已完成）。
2. B-v1 + REM。
3. 按错误类型拆解 B-v1（混淆、漏检、定位，见过/未见过），据此设计方法（斑点重采样、细粒度分类等）。
4. B-v1 补 seed 1、2，得到种子波动。
5. 最优组合跑 3 个种子；补 YOLOv8、plain D-FINE-M 对照（只作对照，不用于更换基线）。

"旧配方 + trainval"对照不跑：它同时差多项设定，不能归因，也不改变任何决定。单项贡献如需写入论文，在 B-v1 上做去掉单项的消融。

目标：多种子均值的 test AP50 超过 SAR-Net 的 90.2。val 在训练集内，模块只能在 test 上比较，因此候选提前列定，失败的也如实报告。

## 3. 服务器命令（旧服务器）

更新代码与环境：

```bash
cd /home/zxy/sar/repos/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export DATA=/home/zxy/sar/datasets/SAR-AIRcraft-1.0
```

一次性准备：确认 trainval 标注（应为 3489 张、13513 个目标），生成类别头初始化权重（新文件，不覆盖原权重）：

```bash
python -c "import json; d=json.load(open('$DATA/COCO/trainval.json')); print(len(d['images']), len(d['annotations']))"
python tools/init_head_from_coco_class.py /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth \
  /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e_airplane7.pth
```

启动（两卡空闲时）：

```bash
nvidia-smi -i 0,1
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e_airplane7.pth
bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_b_canvas1024_132e_b12_old_server.yml
```

启动后检查（重连后同样先设置 `OUT`、激活环境）：

```bash
export OUT=$(ls -dt /home/zxy/sar/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_b_canvas1024_132e_b12_old_server_* | head -n 1); echo "$OUT"
grep -m1 "fixed_schedule" "$OUT/console.log"
grep -m1 "Load model.state_dict" "$OUT/console.log" | cut -c1-400
grep -E "Epoch: \[[0-9]+\]" "$OUT/console.log" | tail -n 1 | grep -oE "Epoch: \[[0-9]+\] +\[[0-9]+/[0-9]+\]|eta: [0-9:]+|time: [0-9.]+|data: [0-9.]+|max mem: [0-9]+"
grep -iE "out of memory|Traceback" "$OUT/console.log" | tail -n 3
```

`Load model.state_dict` 一行里 `unmatched` 只应有 `decoder.anchors`、`decoder.valid_mask`（按 1024 重新生成的缓冲区，尺寸与 COCO 640 不同），类别头不应出现。每轮应为 290 iter（`drop_last`）。

监控：

```bash
tail -f "$OUT/console.log"
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
nvidia-smi -i 0,1
python - "$OUT/log.txt" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
for r in rows[-3:]:
    v = r['test_coco_eval_bbox']
    print(r['epoch'], 'train_loss', round(r['train_loss'], 3), '(val in trainval) AP', round(v[0] * 100, 2), 'AP50', round(v[1] * 100, 2))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

只在进程已停止时恢复：

```bash
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

完成（日志最后为 epoch 131，出现 `final.pth`）后在 test 上评估：

```bash
ls -lh "$OUT"/final.pth
export FINALEVAL="$OUT/test_eval_final"; mkdir "$FINALEVAL"
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 nohup torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/final.pth" --test-only --seed 0 \
  --output-dir "$FINALEVAL" --summary-dir "$FINALEVAL/summary" \
  -u val_dataloader.dataset.img_folder="$DATA" val_dataloader.dataset.ann_file="$DATA/COCO/test.json" \
  > "$FINALEVAL/console.log" 2>&1 < /dev/null &
PID=$!; printf '%s\n' "$PID" > "$FINALEVAL/launcher.pid"; disown "$PID"
tail -f "$FINALEVAL/console.log"
grep -A12 "IoU=0.50:0.95 | area=   all | maxDets=100 ]" "$FINALEVAL/console.log" | head -n 13
```

各类 AP50：

```bash
python - "$FINALEVAL" <<'PY'
import sys, torch
from pathlib import Path
path = next(Path(sys.argv[1]).rglob('*.pth'))
ev = torch.load(path, map_location='cpu', weights_only=False)
p = ev['precision'][0, :, :, 0, -1]  # IoU 0.50, all areas, maxDets 100
names = ['A220', 'A320/321', 'A330', 'ARJ21', 'Boeing737', 'Boeing787', 'other']
for k, n in enumerate(names):
    v = p[:, k]; v = v[v > -1]
    print(n, round(float(v.mean()) * 100, 2))
PY
```

类别顺序按 `test.json` 的 categories；若数量不是 7 或顺序不同，以 `$DATA/COCO/test.json` 中的顺序为准。

## 3.1 新服务器运行 B-v1 + REM

旧服务器 GPU 被占时的单次运行，不改变默认服务器。配置 `deim_hgnetv2_m_sar_aircraft_b_canvas1024_rem_132e_b12_new_server.yml`
与旧服务器版只差路径；新服务器路径见 `AGENTS.md`，初始化权重同样由 `tools/init_head_from_coco_class.py` 生成。
B-v1 基线在旧服务器上运行，比较时注意两台机器的软件版本可能不同；新服务器为 torch 2.1.2、torchvision 0.16.2。

运行：`/root/zxy/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_b_canvas1024_rem_132e_b12_new_server_seed0_20261104_102922_JpWE66`（revision `aef3c03`，GPU 0、1）。

B-v1 基线的新服务器配置为 `deim_hgnetv2_m_sar_aircraft_b_canvas1024_132e_b12_new_server.yml`，只改路径，同为 seed 0，用于检验 REM、A 与 B-v1 的跨服务器比较是否可靠（§1.7）。

## 3.2 错误拆解（B-v1 与之后的每个方法）

`tools/diagnose_detection_errors.py` 只用 CPU，读 `tools/export_analysis_predictions.py` 导出的 test 预测，回答：
AP50 丢在型号混淆、背景误检还是其余（漏检、定位差）；见过和没见过的目标各多少（相对 trainval，匹配器同 `check_split_overlap.py`）；
没见过的飞机是没检出还是型号分数被分散（类别无关 AP50，决定 Q4）。只作诊断，不用于选权重，也不改官方指标。

旧服务器上的 B-v1（`RUN` 换成别的运行目录即可诊断其他方法；导出约占每卡 8 张 1024 图的显存，几分钟）：

```bash
cd /home/zxy/sar/repos/deim
git fetch origin
git switch main
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export DATA=/home/zxy/sar/datasets/SAR-AIRcraft-1.0
export RUN=/home/zxy/sar/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_b_canvas1024_132e_b12_old_server_seed0_20261103_193451_xwONF9
export DIAG=/home/zxy/sar/experiments/diagnostics/errors_bv1_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
mkdir -p "$DIAG"
nvidia-smi -i 0,1
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 nohup torchrun --standalone --nproc_per_node=2 tools/export_analysis_predictions.py \
  -c "$RUN/resolved_config.yml" -r "$RUN/final.pth" --ann-file "$DATA/COCO/test.json" --batch-per-gpu 8 \
  --output "$DIAG/predictions.json.gz" > "$DIAG/export.console.log" 2>&1 < /dev/null &
PID=$!; printf '%s\n' "$PID" > "$DIAG/export.pid"; disown "$PID"
tail -f "$DIAG/export.console.log"
```

显存不够时把 `--batch-per-gpu` 改小。导出结束后：

```bash
CUDA_VISIBLE_DEVICES= nice -n 19 python tools/diagnose_detection_errors.py \
  --predictions "$DIAG/predictions.json.gz" --data-root "$DATA" --output "$DIAG/report"
cat "$DIAG/report/report.md"
```

导出的 AP50 应与 §1.1 的 87.91 一致（只差 float32 与 float16 的舍入）。

漏检原因（`tools/diagnose_missed_targets.py`，只用 CPU，复用同一份预测；回答漏检是没有候选、框不准还是分数低，并按匹配分、切片边缘、对比度、尺寸、场景拆分，生成裁剪拼图）：

```bash
export DIAG=$(ls -dt /home/zxy/sar/experiments/diagnostics/errors_bv1_* | head -n 1); echo "$DIAG"
CUDA_VISIBLE_DEVICES= nice -n 19 python tools/diagnose_missed_targets.py \
  --predictions "$DIAG/predictions.json.gz" --data-root "$DATA" --output "$DIAG/missed"
cat "$DIAG/missed/report.md"
tar -czf "$DIAG/missed.tar.gz" -C "$DIAG" missed
ls -lh "$DIAG/missed.tar.gz"
```

分数证据与误检分组（`tools/diagnose_score_evidence.py`，不训练）：判断"定位确定性重打分"和"强散射硬负样本"两个候选组件是否值得实现。
导出最后一层全部查询的分数和 FDR 边框分布统计（单进程，几分钟；没有空闲显存时用 `--device cpu`），再用 CPU 分析：

```bash
export RUN=/home/zxy/sar/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_b_canvas1024_132e_b12_old_server_seed0_20261103_193451_xwONF9
export DIAG=$(ls -dt /home/zxy/sar/experiments/diagnostics/errors_bv1_* | head -n 1); echo "$DIAG"
nvidia-smi -i 0,1
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python tools/diagnose_score_evidence.py export \
  -c "$RUN/resolved_config.yml" -r "$RUN/final.pth" --ann-file "$DATA/COCO/test.json" \
  --output "$DIAG/query_evidence.npz" > "$DIAG/evidence_export.console.log" 2>&1 < /dev/null &
PID=$!; printf '%s\n' "$PID" > "$DIAG/evidence_export.pid"; disown "$PID"
tail -f "$DIAG/evidence_export.console.log"
```

导出结束后：

```bash
CUDA_VISIBLE_DEVICES= nice -n 19 python tools/diagnose_score_evidence.py analyze \
  --evidence "$DIAG/query_evidence.npz" --data-root "$DATA" --output "$DIAG/evidence"
cat "$DIAG/evidence/report.md"
tar -czf "$DIAG/evidence.tar.gz" -C "$DIAG" evidence
ls -lh "$DIAG/evidence.tar.gz"
```

组件 1 继续的条件：低分的没见过目标对背景和对重复框都明显可分，且 AP50 上限至少高 1。AP50 上限是在 test 上算的，只用于诊断，不得据此设定参数。
拼图 `false_positive_groups.png` 给每个物体编号；把判断为漏标飞机的编号传给 `--unlabelled 1 2 ...`（换新的 `--output`），可得移除这些框后的 AP50。

## 3.3 B-v1 + 斑点重采样

新服务器把命令里的配置换成 `..._speckle_132e_b12_new_server.yml`，路径按 §3.1。其余与 §3 的启动、监控、恢复和 test 评估相同，`OUT` 匹配 `*_b_canvas1024_speckle_132e_b12_*`。

```bash
nvidia-smi -i 0,1
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e_airplane7.pth
bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_b_canvas1024_speckle_132e_b12_old_server.yml
```

启动日志应出现 `### Transform @SpeckleResample ###`，`Policy_ops` 中有 `SpeckleResample`。第 4 轮起看 `data:` 是否明显变长。

## 3.4 B-v1 + 90° 旋转

旧服务器（两卡空闲时）；新服务器把配置换成 `..._rot90_132e_b12_new_server.yml`，路径按 §3.1。其余与 §3 的启动、监控、恢复和 test 评估相同，`OUT` 匹配 `*_b_canvas1024_rot90_132e_b12_*`。

```bash
nvidia-smi -i 0,1
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e_airplane7.pth
bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_b_canvas1024_rot90_132e_b12_old_server.yml
```

启动日志应出现 `### Transform @RandomRot90 ###`，`Policy_ops` 中有 `RandomRot90`。

## 4. 本地验证

Windows Conda `pytorch`（CPU）：`tests/test_sar_protocol_b.py` 检查配置各项；`PadToCanvas` 对 800 与 1500 图补边后框坐标不变、
`orig_size` 为画布、缩放比一致，按画布还原到原图像素误差 <0.01 px，超出画布报错；类别头初始化复制 airplane 行并保留去噪嵌入的填充行。
未在服务器运行；torchvision 版本差异下 `F.pad` 对 PIL 与框的行为只在本地 0.29 上验证。
