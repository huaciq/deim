# BCM 实验交接文档

## 实验概述

**BCM（Boundary Consistency Map）**：边界一致性图模块，用于提升 SAR 小目标的定位质量。

**核心思路**：
- 现有方法（DHQ/PaQ）用 IoU 预测（标量）评估 Query 质量
- 我们用**边界特征图**（空间分布）评估定位质量
- 更适合 SAR 的边界模糊问题

**实现方式**：
- BCM 作为 Encoder 的辅助分支，提取边界特征
- 用辅助损失训练 BCM（让它学会识别边界）
- 间接通过 Encoder 特征影响 Decoder 的定位

**预期效果**：
- 如果 BCM 学到了有用的边界特征，应该提升 AP75（高 IoU 定位）
- 预期涨幅：+0.5~1.5 AP

---

## 服务器启动命令

### 1. 更新代码

```bash
cd /home/zxy/sar/repos/deim
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
conda activate deim_zxy
```

确认拉取到最新 commit `b0a66b3`（"Implement Boundary Consistency Map..."）。

### 2. 启动 BCM 训练

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_bcm/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_bcm.yml "$OUT/config.yml"
nohup env CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_bcm.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
echo $! > "$OUT/launcher.pid" && disown
echo "BCM experiment started: $OUT"
```

**预计时间**：10~12 小时（单卡 4090）

### 3. 监控训练

```bash
# 查看 GPU 占用
nvidia-smi

# 查看实时日志
tail -f /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_bcm/seed0_*/console.log

# 查看训练进度
tail -n 3 /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_bcm/seed0_*/log.txt
```

### 4. 检查中期指标（55 epoch）

```bash
RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_bcm/seed0_*  # 替换为实际目录
sed -n '56p' "$RUN/log.txt" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
print(f\"BCM@55e: AP={d['test_coco_eval_bbox'][0]*100:.1f} AP75={d['test_coco_eval_bbox'][2]*100:.1f} APs={d['test_coco_eval_bbox'][3]*100:.1f}\")
"
```

**对比参考**：
- Baseline（单卡 noaug）@55e: 大约 AP 30~32（根据 120e 结果反推）
- 如果 BCM@55e > baseline@55e + 0.5，说明有效
- 如果 BCM@55e < baseline@55e，立即停止（`kill <PID>`）

### 5. 提取最终结果

```bash
# 最佳 epoch
grep best_stat "$RUN/log.txt" | tail -1

# 完整指标
tail -n 3 "$RUN/log.txt" | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    if 'test_coco_eval_bbox' in d:
        m = d['test_coco_eval_bbox']
        print(f\"AP={m[0]*100:.1f} AP50={m[1]*100:.1f} AP75={m[2]*100:.1f} APs={m[3]*100:.1f} APm={m[4]*100:.1f} APl={m[5]*100:.1f}\")
        break
"
```

---

## 结果对比表

| 模型 | AP | AP75 | APs | 涨幅 | 判断 |
|---|---|---|---|---|---|
| Baseline（noaug） | 57.9 | 56.1 | 50.9 | - | 基线 |
| BCM | ? | ? | ? | ? | 待测 |

**判断标准**：
- **有效**：AP 涨幅 > 0.5，且 AP75 涨幅明显（说明定位质量确实改善）
- **无效**：AP 涨幅 < 0.3，或 AP75 没涨（说明 BCM 没学到有用特征）

---

## 如果 BCM 有效

### 下一步实验
1. **SAQS（形状自适应评分）**：在 BCM 基础上，加入形状信息
2. **BRL（边界约束损失）**：显式约束边界对齐
3. **组合测试**：BCM + SAQS + BRL

### 论文定位
- 核心贡献：边界一致性作为定位质量的评估维度
- 和竞争对手的差异：空间分布 vs 标量 IoU

---

## 如果 BCM 无效

### 可能的原因
1. **边界特征不足**：Encoder 特征分辨率（stride 8）不够精细
2. **辅助损失权重不当**：loss_bcm 权重 0.5 可能太小或太大
3. **GT boundary map 质量差**：line_width=2 可能不合适

### 调试方向
1. **可视化 boundary_map**：看 BCM 是否真的学到了边界
2. **调整损失权重**：试试 loss_bcm: 0.2 或 1.0
3. **改变 line_width**：试试 1 或 3

### 备选方案
- 放弃 BCM，回到"移除增强"作为主要贡献
- 在第二数据集（SSDD/HRSID）上验证增强策略的泛化性
- 调研其他更简单的改进方向

---

## 技术细节

### BCM 模块架构
```
Encoder output (stride 8)
    ↓
BCM (3 层轻量卷积)
    ↓
Boundary Map [B, 1, H, W]
    ↓
BCE Loss vs GT Boundary
```

### GT Boundary Map 生成
- 从 GT boxes（cxcywh）提取边界
- 在 feature map 分辨率（stride 8）上绘制边界线
- line_width=2（feature map pixels）
- 类别不平衡处理：pos_weight = num_neg / num_pos

### 损失配置
```yaml
DEIMCriterion:
  weight_dict: {
    loss_mal: 1,
    loss_bbox: 5,
    loss_giou: 2,
    loss_fgl: 0.15,
    loss_ddf: 1.5,
    loss_bcm: 0.5  # BCM 辅助损失
  }
```

---

## 常见问题

### Q: BCM 会增加多少计算量？
A: 训练时增加约 5%（BCM 分支 + loss 计算）；推理时可以去掉（zero overhead）。

### Q: 为什么不直接用于 Query Selection？
A: 完全集成需要大量修改 Decoder。当前方案通过 Encoder 特征间接影响，改动最小。

### Q: 如果 55 epoch 就看出无效，要继续跑吗？
A: 不用。立即停止，节省时间。BCM 如果有效，应该在早期就能看到涨幅。

### Q: loss_bcm 在 log.txt 里是什么值？
A: 正常应该在 0.1~0.5 之间（BCE loss）。如果 > 1.0，说明训练不稳定。

---

## 总结

BCM 是一个基于边界一致性的辅助监督模块，理论上应该能提升定位质量（AP75）。

**关键指标**：
- AP75 提升 > 1.0：非常有效
- AP75 提升 0.5~1.0：有效
- AP75 提升 < 0.5：效果不明显

等 55 epoch 中期结果出来，立即判断是否继续。如果有效，继续实现 SAQS 和 BRL；如果无效，换方向。
