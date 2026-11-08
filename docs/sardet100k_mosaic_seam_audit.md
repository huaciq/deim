# SARDet-100K：DEIM Mosaic 拼接诊断（Dense O2O 适配性，第 1 步）

日期：2026-11-08。脚本 `tools/sardet100k/mosaic_seam_audit.py`（提交 a93c7f8），新服务器结果在
`/root/zxy/experiments/diagnostics/sardet100k_mosaic_seam_audit_20261108/`（`report.json`、`seams.csv`、`targets.csv`、`gallery/`）。
只看训练集的增强输出，不训练、不涉及 val/test 预测。

## 做法

- 直接用配置里的 `Mosaic` 类和参数（`deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop.yml`：output_size 320、旋转 ±10°、
  平移 0.1、缩放 0.5–1.5、fill 0、不用 cache）重放 4000 个样本，seed 0。把同一次 RandomAffine 作用到象限标签图上，
  因此输出的每个像素都知道来自哪张原图或是填充 0。训练里 Mosaic 只在 epoch [4, 20) 以 0.5 概率使用。
- 物理接缝：相邻象限两侧各 8 px 条带的平均灰度比（灰度 dB，不是后向散射）。
- 自然参照：同一象限里距接缝 32 px 的虚拟接缝，即同一张真实图像内同长度、同方向的边界。
- 目标上下文：每个 GT 外扩一个长边的环，统计其中来自其他象限或填充的比例；≥10% 记为“上下文被拼接污染”。
- 来源分组复用数据集画像的像素匹配缓存（OGSOD、MSAR，其余按图像尺寸）。

## 结果

| 量 | 值 |
| --- | --- |
| 自然虚拟接缝灰度阶跃 P50 / P95 | 0.16 / 1.53 dB |
| 其中 coast 图像内 P50 / P95 | 0.22 / 1.11 dB |
| 物理接缝 P50 / P95 | **6.72 / 18.5 dB** |
| 含跨来源接缝的 Mosaic | 98.0% |
| 来自 ≥3 个来源组的 Mosaic | 68.0% |
| 同一 Mosaic 内象限缩放比 max/min，P50 / P95 | 2.0 / 3.1 |
| 填充 0 占画布 P50 / P75 | 11.5% / 44.6% |
| 每样本目标数：单图 P50 → Mosaic P50 | 1 → 5 |
| Mosaic 目标中上下文被污染 | **39.0%**（其他象限 27.6%，填充 14.8%，跨来源 22.8%） |

接缝类型（share / 阶跃 P50 / 超过自然 P95 的比例）：

| 类型 | share | P50 dB | > 自然 P95 |
| --- | ---: | ---: | ---: |
| land\|sea 跨来源 | 32.3% | 9.83 | 99.2% |
| sea\|sea 跨来源 | 27.2% | 5.41 | 83.3% |
| sea\|sea 同来源组 | 12.7% | 4.54 | 79.6% |
| land\|land 同来源组 | 3.4% | 0.86 | 23.2% |

按类别的上下文污染：harbor 72.9%，aircraft 44.7%，ship 44.4%，bridge 29.2%，tank 21.2%，car 17.7%。

## 解读

1. 阶跃大小本身不是“非自然”的证据：真实河岸/海岸也可能很陡（见 `gallery/natural_coast_reference.jpg`），虚拟接缝
   很少恰好压在海岸线上，所以自然 P95 低估了海岸对比度。真正不存在于真实 SAR 场景的是：长直线边界、
   同为“海”却相差 5 dB 的辐射定标与斑点纹理（sea|sea 同来源组也有 4.5 dB，来源组内部也不一致）、同一画面里 2–3 倍的
   像素尺度差，以及大片填充 0（外观接近平静水面或雷达阴影）。
2. Dense O2O 从 Mosaic 拿到约 5 倍正样本，其中约四成的上下文被拼接改写；harbor、bridge 这类靠上下文定义的类别受影响最大。
3. 这只说明 Mosaic 在 SAR 上产生了非物理的场景结构，**不说明它降低了 AP**。是否有害要靠训练对照判断。

## 限制

- 海/陆标签用固定灰度阈值（平滑后 < 40 的比例），不同来源辐射定标不同：MSAR 的海偏亮、斑点强，会被判成 land
  （例：`gallery/random/00_sample3458.jpg` Q4）；other_512 的港口图有被判成 sea 的。凡是按 sea/land 拆开的数字只能作参考；
  跨来源、阶跃 dB、上下文污染不依赖这个标签。
- 有些来源图本身就是芯片拼图（`natural_coast_reference.jpg` 中 other_512 那张 4×4 船只芯片），它们自带物理接缝。
- 只统计 Mosaic 本身；RandomPhotometricDistort、翻转和 640 Resize 不改变拼接结构，未纳入。
