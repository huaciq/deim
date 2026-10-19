# SPI (Scattering Pattern Isolation) 实验指南

## 创新性说明

### 与现有方法的本质区别

| 维度 | 传统方法 | SPI（我们的方法）|
|---|---|---|
| **特征表示** | RGB/灰度视觉特征 | **SAR散射特性分解** |
| **物理意义** | 无（纯数据驱动） | **显式建模散射机制** |
| **目标定义** | "看起来像目标" | **"散射特性符合目标物理特性"** |
| **杂波抑制** | 学习视觉差异 | **利用散射模式差异** |
| **理论基础** | CNN | **SAR成像物理 + 深度学习** |

### 与相关工作的对比

1. **极化SAR分解**（Cloude-Pottier, Freeman-Durden）
   - 需要多极化数据（HH/HV/VV）
   - 我们：单极化数据，用深度特征近似散射分解

2. **CFAR检测**
   - 手工设计阈值
   - 我们：学习散射先验，自适应调整

3. **注意力机制**（CBAM, SE-Net）
   - 纯数据驱动，无物理意义
   - 我们：散射相似度指导注意力，可解释

4. **多尺度特征融合**（FPN, ASPP）
   - 通用方法，未考虑SAR特性
   - 我们：专门针对SAR散射模式

---

## 技术原理

### SAR 散射物理

SAR 目标检测的本质是**区分不同散射机制**：

```
目标（船舶/桥梁/油罐）
  ├─ 强散射：金属结构的角反射器效应
  ├─ 几何结构：规则形状
  └─ 低随机性：稳定散射

背景杂波（海浪/建筑/植被）
  ├─ 分散散射：体散射/表面散射
  ├─ 无规则结构
  └─ 中等随机性

相干斑噪声
  ├─ 随机散射：乘性噪声
  └─ 高随机性
```

### SPI 模块结构

```
输入特征 (stride-8)
    ↓
局部统计提取（近似散射特性）
    ├─ 均值（散射强度）
    └─ 方差（纹理粗糙度）
    ↓
散射模式分解
    ├─ 强散射概率（目标）
    ├─ 分散散射概率（杂波）
    └─ 随机散射概率（噪声）
    ↓
与学习先验比较
    ├─ 目标先验: [0.7, 0.2, 0.1]
    └─ 杂波先验: [0.2, 0.6, 0.2]
    ↓
计算散射相似度
    ├─ target_similarity（高=目标）
    └─ clutter_similarity（高=杂波）
    ↓
自适应特征门控
    ├─ 增强：target_similarity 高的区域
    └─ 抑制：clutter_similarity 高的区域
    ↓
输出增强特征
```

---

## 预期效果

基于错误分析（Baseline 57.9 AP）：

### 主要改进

1. **背景杂波抑制**
   - 当前：1057 个背景 FP（63.4%）
   - 预期：减少到 600~700（-35~40%）
   - 原因：杂波的分散散射模式会被 SPI 识别并抑制

2. **小目标检测**
   - 当前：1038 个 tiny 漏检（12.4%）
   - 预期：减少到 700~800（-20~30%）
   - 原因：小目标的强散射特征更明显

3. **整体指标**
   - **AP**: 57.9 → **59.5~60.5** (+1.6~2.6)
   - **Precision**: 83.6% → **88~90%**
   - **Tiny Recall**: 87.6% → **90~92%**

---

## 服务器启动命令

```bash
cd /home/zxy/sar/repos/deim
git pull origin main  # 拉取最新代码

# 启动 SPI 实验
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_spi/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")" && mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_spi.yml "$OUT/config.yml"

nohup env CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_spi.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &

echo $! > "$OUT/launcher.pid" && disown
echo "SPI experiment started: $OUT"
```

---

## 监控和验证

### 2 分钟后检查启动状态

```bash
RUN=$(ls -td /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_spi/seed0_* | head -1)
tail -n 30 "$RUN/console.log"
```

**成功标志**：
- 看到 `Epoch: [0]  [ 50/458]` 等训练日志
- Loss 值正常（10-50）
- **没有** OOM 错误

### 中期检查（55 epoch，约 5 小时）

```bash
RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_spi/seed0_*
sed -n '56p' "$RUN/log.txt" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
m = d['test_coco_eval_bbox']
print(f'SPI@55e: AP={m[0]*100:.1f} AP75={m[2]*100:.1f} APs={m[3]*100:.1f}')
"
```

**判断标准**：
- AP@55e > 32 → 有效
- AP@55e < 30 → 效果不佳

### 最终结果（120 epoch）

```bash
RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_spi/seed0_*
tail -n 3 "$RUN/log.txt" | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    if 'test_coco_eval_bbox' in d:
        m = d['test_coco_eval_bbox']
        print(f'SPI Final: AP={m[0]*100:.1f} AP75={m[2]*100:.1f} APs={m[3]*100:.1f}')
        break
"
```

---

## 可视化散射图（可选）

训练完成后，可以可视化散射分解：

```python
# 在 Python 脚本中
import torch
import matplotlib.pyplot as plt

# 加载 checkpoint
checkpoint = torch.load('best_stg1.pth')

# 前向传播一张图
output = model(image)
scatter_probs = output['scatter_probs']  # [1, 3, H, W]
target_sim = output['target_similarity']  # [1, 1, H, W]

# 可视化
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(image[0].cpu(), cmap='gray')
axes[0].set_title('原图')

axes[1].imshow(scatter_probs[0, 0].cpu(), cmap='hot')  # 强散射
axes[1].set_title('强散射（目标）')

axes[2].imshow(scatter_probs[0, 1].cpu(), cmap='hot')  # 分散散射
axes[2].set_title('分散散射（杂波）')

axes[3].imshow(target_sim[0, 0].cpu(), cmap='jet')
axes[3].set_title('目标相似度')

plt.savefig('spi_visualization.png')
```

---

## 论文中的创新性陈述

> **Scattering Pattern Isolation (SPI)**: Unlike conventional CNNs that treat SAR images as grayscale photographs, SPI explicitly decomposes features into three scattering components—strong (targets), diffuse (clutter), and random (speckle)—inspired by polarimetric SAR decomposition. By comparing spatial patterns with learned target/clutter priors, SPI adaptively enhances target-like scattering while suppressing clutter, achieving physics-informed feature selection.

**关键创新点**：
1. 首次在单极化SAR检测中显式建模散射分解
2. 用深度学习近似极化SAR的物理分解
3. 可学习的散射先验替代手工CFAR阈值
4. 可解释性：散射图直观展示物理过程

---

**现在可以启动实验了！**
