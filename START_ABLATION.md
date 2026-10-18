# 消融实验快速启动指南

所有代码已实现并推送到 GitHub（commit `d341a5d`）。

---

## 实验概览

基于错误分析设计的三个模块：

| 模块 | 针对问题 | 预期效果 |
|---|---|---|
| **SCAS** | 63.4% 背景误检 | Precision 83.6% → 90%, AP +1.5~2 |
| **DPM** | 12.4% 小目标漏检 | Tiny Recall 87.6% → 92%, AP +1~1.5 |
| **BCM** | 36.6% 定位误差 | AP75 +0.9（已验证） |

**目标**：Baseline 57.9 → Full 61.5~62

---

## 快速启动（直接复制命令）

### 步骤 1：更新代码

```bash
cd /home/zxy/sar/repos/deim
git pull origin main  # 拉取最新代码（d341a5d）
conda activate deim_zxy
```

---

### 步骤 2：启动 SCAS 实验（优先）

```bash
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_scas/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")" && mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_scas.yml "$OUT/config.yml"
nohup env CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_scas.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
echo $! > "$OUT/launcher.pid" && disown
echo "SCAS started: $OUT"
```

**预计时间**：10~12 小时

---

### 步骤 3：监控训练

```bash
# 实时日志
tail -f /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_scas/seed0_*/console.log

# 最新 3 行
tail -n 3 /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_scas/seed0_*/log.txt

# 中期检查（55 epoch，约 5 小时后）
RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_scas/seed0_*
sed -n '56p' "$RUN/log.txt" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
m = d['test_coco_eval_bbox']
print(f'SCAS@55e: AP={m[0]*100:.1f} AP75={m[2]*100:.1f} APs={m[3]*100:.1f}')
"
```

**判断标准**：
- AP@55e > 31.5 → 有效，继续
- AP@55e < 30.5 → 无效，停止并调试

---

### 步骤 4：根据 SCAS 结果决定下一步

#### 如果 SCAS 有效（AP > 59）

启动 DPM：

```bash
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_dpm/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")" && mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_dpm.yml "$OUT/config.yml"
nohup env CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_dpm.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
echo $! > "$OUT/launcher.pid" && disown
echo "DPM started: $OUT"
```

#### 如果 SCAS + DPM 都有效

启动 Full：

```bash
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_full/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")" && mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_full.yml "$OUT/config.yml"
nohup env CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_full.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
echo $! > "$OUT/launcher.pid" && disown
echo "Full started: $OUT"
```

---

## 提取最终结果

```bash
# SCAS
RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_scas/seed0_*
tail -n 3 "$RUN/log.txt" | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    if 'test_coco_eval_bbox' in d:
        m = d['test_coco_eval_bbox']
        print(f'SCAS: AP={m[0]*100:.1f} AP75={m[2]*100:.1f} APs={m[3]*100:.1f}')
        break
"

# DPM
RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_dpm/seed0_*
tail -n 3 "$RUN/log.txt" | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    if 'test_coco_eval_bbox' in d:
        m = d['test_coco_eval_bbox']
        print(f'DPM: AP={m[0]*100:.1f} AP75={m[2]*100:.1f} APs={m[3]*100:.1f}')
        break
"

# Full
RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_full/seed0_*
tail -n 3 "$RUN/log.txt" | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    if 'test_coco_eval_bbox' in d:
        m = d['test_coco_eval_bbox']
        print(f'Full: AP={m[0]*100:.1f} AP75={m[2]*100:.1f} APs={m[3]*100:.1f}')
        break
"
```

---

## 预期消融表

| Method | AP | AP75 | APs | 说明 |
|---|---|---|---|---|
| Baseline | 57.9 | 56.1 | 50.9 | 移除增强 |
| +SCAS | 59.5 | 57.5 | 51.5 | 背景抑制 |
| +DPM | 59.0 | 57.0 | 53.0 | 细节保留 |
| +BCM | 58.3 | 57.0 | 51.5 | 边界监督（已跑） |
| **+Full** | **61.5** | **59.0** | **54.0** | 三者结合 |

---

## 常见问题

### Q: 启动报错 "p2_channels not set"

A: DPM 需要 P2 feature，但配置文件已经设置了。如果报错，检查 backbone 是否正确输出 4 个特征。

### Q: CUDA out of memory

A: 降低 batch size（配置文件里改 `total_batch_size: 32` → `24`）。

### Q: 如何停止训练？

A: `kill $(cat /path/to/launcher.pid)`

---

**详细文档**：[docs/ablation_experiments_guide.md](docs/ablation_experiments_guide.md)

**现在可以启动 SCAS 实验了！**把最终结果发给我，我帮你分析。
