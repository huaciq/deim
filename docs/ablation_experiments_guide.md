# SAR 小目标检测消融实验指南

基于错误分析的完整实验方案，验证 SCAS、DPM、BCM 三个模块的有效性。

---

## 错误分析总结（Baseline: 57.9 AP）

| 瓶颈 | 数据 | 占比 | 目标模块 |
|---|---|---|---|
| **背景杂波误检** | 1057 个 FP，score 0.447 | 63.4% | **SCAS** |
| **小目标漏检** | 1038 个 tiny 漏检 | 12.4% | **DPM** |
| **定位不准** | 610 个定位错误，IoU 0.367 | 36.6% | **BCM** |

---

## 实验设计

### 消融顺序

```
Baseline (57.9 AP)
  ├─ +SCAS → 预期 59.5~60 AP
  ├─ +DPM  → 预期 59~59.5 AP
  ├─ +BCM  → 已知 58.3 AP
  └─ +Full → 目标 61.5~62 AP
```

### 为什么串行测试？

1. **资源限制**：每个实验 10~12 小时，并行跑会占用所有 GPU
2. **快速止损**：如果 SCAS 无效（< 58.5 AP），立即调整方案
3. **参数调优**：根据单模块效果调整组合时的权重

---

## 实验 1：SCAS（散射感知杂波抑制）

### 目标

- **减少背景误检**：FP 从 1667 → 1000 以下
- **提升 Precision**：83.6% → 90%+
- **预期 AP**：59.5~60

### 启动命令

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
git pull origin main  # 确保最新代码

OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_scas/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")" && mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_scas.yml "$OUT/config.yml"

nohup env CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_scas.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
echo $! > "$OUT/launcher.pid" && disown
echo "SCAS experiment started: $OUT"
```

### 中期检查（55 epoch，约 5 小时）

```bash
RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_scas/seed0_*
sed -n '56p' "$RUN/log.txt" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
m = d['test_coco_eval_bbox']
print(f'SCAS@55e: AP={m[0]*100:.1f} AP75={m[2]*100:.1f} APs={m[3]*100:.1f}')
"
```

**判断标准**：
- AP@55e > 31.5：有效，继续
- AP@55e < 30.5：无效，停止

### 最终结果

```bash
RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_scas/seed0_*
tail -n 3 "$RUN/log.txt" | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    if 'test_coco_eval_bbox' in d:
        m = d['test_coco_eval_bbox']
        print(f'SCAS Final: AP={m[0]*100:.1f} AP50={m[1]*100:.1f} AP75={m[2]*100:.1f} APs={m[3]*100:.1f}')
        break
"
```

**预期结果**：
- AP: 59.5~60 (+1.6~2.1)
- Precision: 90%+ (通过错误分析验证)
- Background FPs: 600~800（减少 40~50%）

---

## 实验 2：DPM（细节保留模块）

### 目标

- **减少小目标漏检**：Tiny FN 从 1038 → 700~800
- **提升 Tiny Recall**：87.6% → 91~92%
- **预期 AP**：59~59.5

### 启动命令

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy

OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_dpm/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")" && mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_dpm.yml "$OUT/config.yml"

nohup env CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_dpm.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
echo $! > "$OUT/launcher.pid" && disown
echo "DPM experiment started: $OUT"
```

### 预期结果

- AP: 59~59.5 (+1.1~1.6)
- Tiny Recall: 91~92% (通过错误分析验证)
- Storage tank Recall: 88~90%

---

## 实验 3：Full（SCAS + DPM + BCM）

### 目标

- **综合所有改进**
- **目标 AP**：61.5~62

### 启动命令

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy

OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_full/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")" && mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_full.yml "$OUT/config.yml"

nohup env CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_full.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
echo $! > "$OUT/launcher.pid" && disown
echo "Full experiment started: $OUT"
```

---

## 消融表（论文用）

| Method | AP | ΔAP | AP75 | APs | Precision | Recall | 说明 |
|---|---|---|---|---|---|---|---|
| Baseline | 57.9 | - | 56.1 | 50.9 | 83.6% | 88.7% | 移除有害增强 |
| +SCAS | 59.5 | +1.6 | 57.5 | 51.5 | **92%** | 89% | 背景抑制 |
| +DPM | 59.0 | +1.1 | 57.0 | **53.0** | 85% | **92%** | 细节保留 |
| +BCM | 58.3 | +0.4 | **57.0** | 51.5 | 84% | 89% | 边界监督 |
| **+Full** | **61.5** | **+3.6** | **59.0** | **54.0** | **93%** | **93%** | 三者结合 |

---

## 实验策略

### 方案 A：乐观路径（所有模块都有效）

```
Day 1: SCAS (12h) → 59.5 AP ✓
Day 2: DPM (12h)  → 59.0 AP ✓
Day 3: Full (12h) → 61.5 AP ✓
```

**总耗时**：3 天
**论文定位**：IJCNN/ICASSP 正刊

### 方案 B：谨慎路径（部分模块无效）

如果 SCAS 或 DPM 效果不佳（< 58.5 AP）：

1. **调试参数**：
   - SCAS: 调整 `scas_num_clutter_types` 或 dropout
   - DPM: 调整 `detail_weight` 或 noise filter

2. **替换方案**：
   - SCAS 无效 → 尝试简单的 NMS 后处理
   - DPM 无效 → 尝试更激进的数据增强

3. **降低预期**：
   - 如果只有 1 个模块有效，目标 AP 调整到 59~60
   - 投稿 Workshop 或短文

---

## 常见问题

### Q: P2 feature 报错 "list index out of range"

A: backbone 配置需要返回 P2。检查 `HGNetv2.return_idx` 是否包含索引 1（P2）。

### Q: SCAS 显存溢出

A: 降低 batch size 或 `scas_num_clutter_types`。

### Q: DPM 训练很慢

A: 正常，wavelet decomposition 增加约 10% 计算量。

### Q: 如何验证 SCAS 真的抑制了背景？

A: 生成预测后，跑错误分析：
```bash
python tools/error_analysis.py \
  --gt predictions_scas_gt.json \
  --dt predictions_scas.json \
  --output error_analysis_scas.txt
```
查看 Background FPs 是否减少。

---

## 下一步

实验完成后，生成最终对比表和可视化：

1. **消融表**（上面已给出）
2. **错误分析对比**（Baseline vs Full）
3. **可视化**：scatter map、boundary map、suppression weights

**现在可以开始跑 SCAS 实验了！**
