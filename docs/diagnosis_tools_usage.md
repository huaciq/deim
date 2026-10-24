# 诊断工具使用指南

完整的错误分析和 Query 诊断流程，帮助找到模型的具体问题。

---

## 步骤 1：生成预测结果

在服务器上运行，生成验证集的预测结果。

### 命令

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy

# 使用最好的 checkpoint（例如 Baseline noaug 57.9 AP）
CHECKPOINT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_noaug/seed0_*/best_stg1.pth
CONFIG=configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_noaug.yml

python tools/generate_predictions.py \
  -c $CONFIG \
  -r $CHECKPOINT \
  -o predictions_baseline.json
```

**输出**：
- `predictions_baseline.json`：COCO 格式的预测结果
- `predictions_baseline_gt.json`：对应的 GT 标注（方便后续分析）

**预计时间**：5~10 分钟

---

## 步骤 2：错误分析

分析漏检（FN）、误检（FP）、定位错误，按类别和尺度统计。

### 命令

```bash
python tools/error_analysis.py \
  --gt predictions_baseline_gt.json \
  --dt predictions_baseline.json \
  --iou 0.5 \
  --output error_analysis_baseline.txt
```

### 输出示例

```
================================================================================
DETECTION ERROR ANALYSIS
================================================================================

IoU Threshold: 0.5

Total GT annotations: 5234
Total predictions: 5812

True Positives (TP): 4821
False Positives (FP): 991
False Negatives (FN): 413

Precision: 0.830
Recall: 0.921
F1 Score: 0.873

--------------------------------------------------------------------------------
PER-CLASS ANALYSIS
--------------------------------------------------------------------------------
Category             GT Count     TP       FP       FN       Precision    Recall      
--------------------------------------------------------------------------------
storage_tank         2145         1876     312      269      0.857        0.875       
bridge               1823         1701     198      122      0.896        0.933       
harbor               1266         1244     481      22       0.721        0.983       

--------------------------------------------------------------------------------
SIZE-BASED ANALYSIS
--------------------------------------------------------------------------------
Size Bin        Total GT     TP       FN       Recall       Miss Rate   
--------------------------------------------------------------------------------
tiny            3567         3156     411      0.885        0.115       
small           1523         1521     2        0.999        0.001       
medium          144          144      0        1.000        0.000       

--------------------------------------------------------------------------------
LOCALIZATION ERROR ANALYSIS
--------------------------------------------------------------------------------
Total localization errors: 456
  (Detections with 0.1 <= IoU < 0.5)
  Mean IoU: 0.382
  Median IoU: 0.391

  Per-class localization errors:
    storage_tank: 198 errors, mean IoU = 0.375
    bridge: 142 errors, mean IoU = 0.387
    harbor: 116 errors, mean IoU = 0.393

--------------------------------------------------------------------------------
BACKGROUND FALSE POSITIVE ANALYSIS
--------------------------------------------------------------------------------
Total background FPs: 535
  (Detections with max IoU < 0.1)
  Mean score: 0.624

  Per-class background FPs:
    storage_tank: 114
    bridge: 56
    harbor: 365

================================================================================
KEY FINDINGS
================================================================================
1. Worst-performing category: storage_tank (Recall: 0.875)
2. Most problematic size: tiny (Miss rate: 0.115)
3. False positive breakdown:
   - Localization errors: 46.0% (456 / 991)
   - Background FPs: 54.0% (535 / 991)
   → Main issue: BACKGROUND CLUTTER (false detections in empty areas)
================================================================================
```

### 关键指标解读

| 指标 | 说明 | 如果高则说明 |
|---|---|---|
| **Recall（召回率）** | TP / (TP + FN) | 漏检少，检测能力强 |
| **Precision（精度）** | TP / (TP + FP) | 误检少，预测可靠 |
| **Localization errors** | IoU 在 0.1~0.5 的 FP | 定位不准确（框不够精确） |
| **Background FPs** | IoU < 0.1 的 FP | 背景杂波误检（框在空白区域） |

### 决策树

**如果 "Miss rate (tiny)" > 15%**：
→ 小目标漏检严重，需要**特征增强**（细节保留 / 多尺度融合）

**如果 "Localization errors" > 50% of FP**：
→ 定位不准确是主要问题，需要**定位优化**（边界约束 / 回归改进）

**如果 "Background FPs" > 50% of FP**：
→ 背景杂波是主要问题，需要**背景抑制**（注意力机制 / 对比学习）

---

## 步骤 3：Query 诊断

分析 Query Selection 的质量，找出高质量 Query 排名是否靠后。

### 命令

```bash
python tools/query_diagnosis.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_noaug.yml \
  -r /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_noaug/seed0_*/best_stg1.pth \
  -o query_diagnosis_baseline.txt
```

### 输出示例

```
================================================================================
QUERY DIAGNOSIS REPORT
================================================================================

OVERALL QUERY RECALL:
--------------------------------------------------------------------------------
Metric                    Top-100      Top-300      Top-500      Top-900     
--------------------------------------------------------------------------------
QR @ IoU=0.30       95.2%        98.1%        99.0%        99.5%       
QR @ IoU=0.50       88.3%        92.7%        94.8%        96.2%       
QR @ IoU=0.70       62.5%        68.4%        72.1%        76.8%       
QR @ IoU=0.75       48.7%        52.6%        56.3%        61.2%       
QR @ IoU=0.90       12.4%        15.3%        18.7%        23.1%       

Best-query rank:
  Mean: 156.3
  Median: 92.0

--------------------------------------------------------------------------------
PER-CLASS QUERY RECALL @ IoU=0.5 (Top-300):
--------------------------------------------------------------------------------
storage_tank         QR@0.5:   89.2%   Mean rank:  178.5
bridge               QR@0.5:   94.8%   Mean rank:  124.7
harbor               QR@0.5:   95.1%   Mean rank:  142.3

--------------------------------------------------------------------------------
PER-SIZE QUERY RECALL @ IoU=0.5 (Top-300):
--------------------------------------------------------------------------------
tiny                 QR@0.5:   91.5%   Mean rank:  165.2
small                QR@0.5:   96.8%   Mean rank:  121.4
medium               QR@0.5:   98.2%   Mean rank:   85.6

================================================================================
KEY FINDINGS:
================================================================================
1. Top-300 queries capture 92.7% of GTs at IoU=0.5
2. Top-300 queries capture 52.6% of GTs at IoU=0.75
3. Gap: 40.1 percentage points

→ DIAGNOSIS: Objects are DETECTED but LOCALIZATION is poor
   Recommendation: Focus on improving box regression quality
================================================================================
```

### 关键指标解读

| 指标 | 说明 | 正常值 | 如果低则说明 |
|---|---|---|---|
| **QR@0.5 (Top-300)** | Top-300 Query 捕获 GT 的比例（IoU≥0.5） | > 90% | Query Selection 弱，目标被遗漏 |
| **QR@0.75 (Top-300)** | Top-300 Query 捕获 GT 的比例（IoU≥0.75） | > 70% | 定位质量差，框不够精确 |
| **QR@0.5 - QR@0.75 Gap** | 两者差距 | < 25% | 定位质量是瓶颈 |
| **Mean best-query rank** | 最佳 Query 的平均排名 | < 150 | 高质量 Query 排名靠后 |

### 决策树

**如果 QR@0.5 (Top-300) < 90%**：
→ 很多目标根本没被 Query Selection 找到，需要**改进 Query 初始化或特征提取**

**如果 QR@0.5 > 90% 且 QR@0.75 < 60%**：
→ **目标能检出但定位不准**，需要**改进回归或边界约束**（这是最常见的情况）

**如果 Mean rank > 200**：
→ 高质量 Query 排名太靠后，需要**改进 Query Scoring**（这就是 BCM/SAQS 的动机）

---

## 步骤 4：综合诊断报告

结合错误分析和 Query 诊断，生成完整报告。

### 模板

```
模型：Baseline (noaug, 57.9 AP)

【错误分析】
- 总 GT：5234，总预测：5812
- Recall: 0.921，Precision: 0.830
- 最差类别：storage_tank (Recall 0.875)
- 最差尺度：tiny (Miss rate 11.5%)
- FP 主要问题：背景杂波 (54%)

【Query 诊断】
- QR@0.5 (Top-300): 92.7%
- QR@0.75 (Top-300): 52.6%
- Gap: 40.1%
- Mean rank: 156.3

【结论】
1. 主要瓶颈：定位精度差（QR@0.5 高但 QR@0.75 低）
2. 次要问题：小目标漏检（tiny miss rate 11.5%）
3. 背景误检也需要关注（54% FP 来自背景）

【建议方向】
优先级 1：改进定位精度（多尺度回归 / 边界约束）
优先级 2：特征增强（细节保留）
优先级 3：背景抑制（注意力机制）
```

---

## 步骤 5：对比不同模型

如果你有多个模型（例如 Baseline 和 BCM），可以对比分析。

```bash
# Baseline
python tools/error_analysis.py \
  --gt predictions_baseline_gt.json \
  --dt predictions_baseline.json \
  --output error_analysis_baseline.txt

# BCM
python tools/generate_predictions.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_bcm.yml \
  -r /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_bcm/seed0_*/best_stg1.pth \
  -o predictions_bcm.json

python tools/error_analysis.py \
  --gt predictions_bcm_gt.json \
  --dt predictions_bcm.json \
  --output error_analysis_bcm.txt
```

### 对比分析

| 指标 | Baseline | BCM | 变化 | 说明 |
|---|---|---|---|---|
| Recall | 0.921 | 0.925 | +0.004 | 漏检略微减少 |
| Precision | 0.830 | 0.832 | +0.002 | 误检略微减少 |
| Localization errors | 456 | 423 | -33 | **定位错误减少** ✓ |
| Background FPs | 535 | 542 | +7 | 背景误检略微增加 |

**结论**：BCM 主要改善了定位精度（Localization errors 下降），符合设计初衷。

---

## 常见问题

### Q: 生成预测结果时报错 "module 'cfg' has no attribute 'postprocessor'"

A: 某些配置文件没有 postprocessor，修改 `generate_predictions.py` 的第 44 行：

```python
# 改成直接用 model 输出
orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
results = []
for i in range(len(outputs['pred_boxes'])):
    # 手动 post-process
    scores, labels = outputs['pred_logits'][i].sigmoid().max(dim=-1)
    keep = scores > 0.3  # Score threshold
    results.append({
        'boxes': outputs['pred_boxes'][i][keep],
        'scores': scores[keep],
        'labels': labels[keep]
    })
```

### Q: 错误分析的 IoU 阈值应该用多少？

A: 建议用 **0.5**（COCO 标准）。如果想看高精度定位，可以额外跑一次 `--iou 0.75`。

### Q: Query 诊断需要多长时间？

A: 约 10~15 分钟（遍历验证集，计算所有 Query 和 GT 的 IoU）。

---

**现在你可以开始跑诊断了！按步骤 1→2→3 依次执行，把结果发给我，我帮你分析具体问题。**

## 定位上界诊断（localization_ceiling.py）

用途：在写新模块之前，先量化 AP 的损失该归给哪一类原因，避免又一次凭直觉设计模块。
同一组检测结果做四种受控替换，所有变体检测集不变，AP 差异只来自一个因素。

| 变体 | 含义 | 对应可改进项 |
|---|---|---|
| `AP` | 原始检测 | 参考值 |
| `AP_GTBOX` | 匹配上的框换成 GT 框 | **纯定位精度**的上界 |
| `AP_TPFP` | 匹配上的保留，其余分数置 0 | **前后景分离**的上界 |
| `AP_IQS` | 分数 × 真实 IoU | **分数排序质量**的上界 |
| `AP_BOTH` | 两者都做 | 联合上界 |

判读：
- `AP_GTBOX - AP` 大 → 定位是瓶颈，做精修/高分辨率采样
- `AP_TPFP - AP` 大 → 误检是瓶颈，做背景抑制
- `AP_IQS - AP` 大 → 分数不反映框质量，做质量感知排序

必须关掉 NMS 并把分数阈值放很低，否则检测集被裁剪，上界不成立。

    RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_noaug_b24/seed0
    cd /home/zxy/sar/repos/deim

    python tools/generate_predictions.py \
      -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_noaug_b24.yml \
      -r "$RUN/best_stg1.pth" \
      -o predictions_b24.json \
      --score-thr 0.001 --no-nms

    python tools/localization_ceiling.py \
      --gt /data2/zxy/sar/datasets/OGSOD-1.0/sar/RTDETR_COCO/val.json \
      --dt predictions_b24.json \
      --iou-thr 0.5 \
      --output ceiling_b24.txt
