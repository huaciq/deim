# F1/F2/F3 基础调参实验交接

## 背景

E1（NWD）和 E2（HWD P2 注入）两个模块实验均失败：
- **E1-NWD**：55 epoch 时比 baseline 低 1.4 个 AP 点（28.6 vs 30.0），完全失效
- **E2-HWD**：120 epoch 比 baseline 仅高 0.3 个 AP 点（53.0 vs 52.7），且中等目标 APm 掉了 6.5 个点

**新假设**：问题不在缺少模块，而在基础设置不适配 SAR 小目标。

这三个实验测试**增强策略、回归尺度、学习率**是否与 OGSOD 数据特性不匹配。

---

## 实验概览

| 编号 | 配置文件 | 改动 | 假设 | 预期涨幅 |
|---|---|---|---|---|
| **F1** | `deim_hgnetv2_s_ogsod_120e_noaug.yml` | 移除 RandomZoomOut 和 RandomIoUCrop | 这两个增强会把小目标缩得更小或裁掉 | +1~2 AP |
| **F2** | `deim_hgnetv2_s_ogsod_120e_regscale2.yml` | reg_scale 从 4 降到 2 | D-FINE 默认 4 对 11px 中位数目标太粗糙 | +0.5~1.5 AP |
| **F3** | `deim_hgnetv2_s_ogsod_120e_lowlr.yml` | backbone lr 从 0.0002 降到 0.0001<br>warmup 从 1000 增到 2000 | 小目标对 LR 敏感，当前设置过快 | +0.5~1 AP |

**执行顺序**：
1. **优先跑 F1**（最可能有效）
2. 如果 F1 涨幅 > 0.5，在 F1 基础上叠加 F2 和 F3
3. 如果 F1 无效，直接测 F2

每个实验约 6 小时，用两张 4090。

---

## 服务器更新与启动

### 1. 更新代码

```bash
cd /home/zxy/sar/repos/deim
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
conda activate deim_zxy
```

确认拉取到最新 commit `9063159`（"Add F1/F2/F3 baseline tuning..."）。

### 2. 启动 F1（移除有害增强）

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_f1_noaug/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_noaug.yml "$OUT/config.yml"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_noaug.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
disown
echo "F1-noaug started: $OUT"
```

### 3. 启动 F2（降低回归尺度）

**等 F1 跑到 55 epoch 左右、确认是否有效后再决定**：
- 如果 F1@55e > baseline@55e（30.0）且涨幅 > 0.5，让 F1 继续跑到 120e
- 同时启动 F2：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_f2_regscale2/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_regscale2.yml "$OUT/config.yml"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_regscale2.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
disown
echo "F2-regscale2 started: $OUT"
```

### 4. 启动 F3（降低学习率）

**等 F1 和 F2 的 55 epoch 结果都出来后**，如果：
- F1 有效但不够（涨幅 0.5~1），F2 也有效 → 在 F1+F2 基础上叠加 F3（需要新建组合配置）
- F1 和 F2 都无效 → 单独测 F3：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_f3_lowlr/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_lowlr.yml "$OUT/config.yml"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_lowlr.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
disown
echo "F3-lowlr started: $OUT"
```

---

## 监控与提前停止

### 查看 55 epoch 的中期指标

```bash
# 查看 log.txt 第 55 行（epoch 从 0 开始，所以第 56 行）
RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_f1_noaug/seed0_YYYYMMDD_HHMMSS
sed -n '56p' "$RUN/log.txt" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
print(f\"F1@55e: AP={d['test_coco_eval_bbox'][0]*100:.1f} APs={d['test_coco_eval_bbox'][3]*100:.1f}\")
"
```

**对比基线**：
- Baseline@55e: AP 30.0, APs 22.2
- 如果 F1@55e < 30.5，立即停掉（`kill <PID>`）
- 如果 F1@55e > 31.0，很可能有效，继续跑到 120e

### 最终结果提取

```bash
# 查看最佳 epoch
grep best_stat "$RUN/log.txt" | tail -1

# 提取所有指标
tail -n 3 "$RUN/log.txt" | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    if 'test_coco_eval_bbox' in d:
        m = d['test_coco_eval_bbox']
        print(f\"AP={m[0]*100:.1f} AP50={m[1]*100:.1f} AP75={m[2]*100:.1f} APs={m[3]*100:.1f} APm={m[4]*100:.1f} APl={m[5]*100:.1f}\")
"
```

---

## 判断标准

| 结果 | 判断 | 下一步 |
|---|---|---|
| F1 涨 > 1.5 | ✓ 增强策略是主因 | 在 F1 基础上叠加 F2，写论文时强调数据增强适配 |
| F1 涨 0.5~1.5 | △ 有改善但不够 | 叠加 F2 和 F3 |
| F1 涨 < 0.5 | ✗ 无效 | 测 F2，如果也无效考虑换 baseline 或数据集 |

**关键指标**：
- **总 AP**：必须比 52.7 高 0.5 以上
- **APs**：重点，必须比 46.1 高
- **AP75**：如果 F2 有效，这个应该涨得比 AP50 多

---

## 如果三个都无效

回到根本性问题：

### 选项 A：换成原版 D-FINE（不用 DEIM 的 MAL）

DEIM 的 MAL 损失在 COCO 上有效，但可能不适合 SAR 小目标。测试原版 D-FINE：

```bash
# 需要新建配置，改用 dfine_hgnetv2_s_coco.yml 作为 base
# 把 DEIMCriterion 的 loss_mal 改回 loss_vfl
```

### 选项 B：换数据集验证

在另一个 SAR 小目标数据集（SSDD 或 HRSID）上测，看问题是方法的还是 OGSOD 标注质量的。

### 选项 C：换 baseline

试试 RT-DETRv2 或 DEIMv2，看是否 D-FINE 的架构本身不适合 SAR 小目标。

---

## 对比表（占位）

| 模型 | AP | AP50 | AP75 | APs | APm | APl | 备注 |
|---|---|---|---|---|---|---|---|
| S-baseline-120e | 52.7 | 87.7 | 50.9 | 46.1 | 80.4 | 81.0 | 基线 |
| M-baseline-120e | 56.9 | 90.8 | 56.2 | 50.5 | 80.4 | 84.0 | 目标：S 调参后超过这个 |
| E1-NWD | 28.6@55e | - | - | - | - | - | ✗ 失败 |
| E2-HWD | 53.0 | 88.2 | 51.1 | 46.6 | 73.9 | 79.7 | △ 涨 0.3，无效 |
| F1-noaug | ? | ? | ? | ? | ? | ? | 移除有害增强 |
| F2-regscale2 | ? | ? | ? | ? | ? | ? | 降低回归尺度 |
| F3-lowlr | ? | ? | ? | ? | ? | ? | 降低学习率 |

把 F1/F2/F3 的 55 epoch 和 120 epoch 结果填进这个表，重点看哪个涨幅最大。

---

## 关于文献

**现在依然不需要收集**。如果 F1 到 F3 有一个涨幅超过 1.5 个点，说明基础设置是主因，到时候写论文重点讲数据特性分析和针对性调整。

只有当所有基础调参都试过、还是卡在 53 左右，才考虑换方向或引入新模块。
