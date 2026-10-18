# BCM 实验启动指南

## ✅ 代码已完成并测试通过

**Commit**: `b0a66b3` - "Implement Boundary Consistency Map (BCM) for SAR small object detection"

**本地测试结果**：
```
[OK] BCM output shape: torch.Size([2, 1, 80, 80])
[OK] GT boundary shape: torch.Size([2, 1, 80, 80])
[OK] Loss computation: 0.9823
[OK] All tests passed! BCM is ready.
```

---

## 🚀 服务器启动命令（直接复制）

### 步骤 1：更新代码

```bash
cd /home/zxy/sar/repos/deim
git fetch origin && git pull --ff-only origin main
conda activate deim_zxy
git log --oneline -1  # 确认是 b0a66b3
```

### 步骤 2：启动训练

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_bcm/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")" && mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_bcm.yml "$OUT/config.yml"
nohup env CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_bcm.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
echo $! > "$OUT/launcher.pid" && disown
echo "BCM started: $OUT"
```

**预计时间**：10~12 小时

---

## 📊 监控命令

### 查看实时日志

```bash
tail -f /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_bcm/seed0_*/console.log
```

### 查看最新 3 行训练日志

```bash
tail -n 3 /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_bcm/seed0_*/log.txt
```

### 检查 55 epoch 中期结果（约 5 小时后）

```bash
RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_bcm/seed0_*
sed -n '56p' "$RUN/log.txt" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
m = d['test_coco_eval_bbox']
print(f'BCM@55e: AP={m[0]*100:.1f} AP75={m[2]*100:.1f} APs={m[3]*100:.1f}')
"
```

**判断标准**：
- 如果 AP@55e > 31.0：有效，继续跑
- 如果 AP@55e < 30.5：无效，立即停止（`kill $(cat "$RUN/launcher.pid")`）

---

## 📈 最终结果提取

```bash
RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_bcm/seed0_*

# 最佳 epoch
grep best_stat "$RUN/log.txt" | tail -1

# 完整指标
tail -n 3 "$RUN/log.txt" | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    if 'test_coco_eval_bbox' in d:
        m = d['test_coco_eval_bbox']
        print(f'AP={m[0]*100:.1f} AP50={m[1]*100:.1f} AP75={m[2]*100:.1f} APs={m[3]*100:.1f} APm={m[4]*100:.1f} APl={m[5]*100:.1f}')
        break
"
```

---

## 🎯 对比参考

| 模型 | AP | AP75 | APs | 判断 |
|---|---|---|---|---|
| Baseline（noaug） | 57.9 | 56.1 | 50.9 | 基线 |
| **BCM（目标）** | **58.5+** | **57.0+** | **51.5+** | 需验证 |

**有效标准**：
- AP 涨幅 > 0.5
- AP75 涨幅 > 0.5（关键指标，说明定位质量确实改善）

---

## 🔧 如果出现问题

### 问题 1：CUDA out of memory

**解决**：降低 batch size（当前 32，降到 24 或 16）

```yaml
# 修改配置文件
train_dataloader:
  total_batch_size: 16  # 从 32 改成 16
```

### 问题 2：loss_bcm 是 NaN

**原因**：GT boundary map 生成有问题或类别不平衡太严重

**解决**：检查 console.log 里的错误信息

### 问题 3：训练速度比 baseline 慢很多

**原因**：BCM 分支增加了计算量

**预期**：慢 5~10%（每 epoch 从 5.5 分钟增加到 5.8~6 分钟）

---

## 📝 完整技术说明

详见：[docs/bcm_experiment_handoff.md](docs/bcm_experiment_handoff.md)

---

## ✅ 检查清单

启动前确认：

- [ ] 代码已更新到 commit `b0a66b3`
- [ ] Conda 环境 `deim_zxy` 已激活
- [ ] COCO 预训练权重存在：`/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth`
- [ ] GPU 0 空闲（`nvidia-smi`）
- [ ] 实验目录有足够空间（约 10 GB）

启动后：

- [ ] 5 分钟后检查 `console.log` 有无报错
- [ ] 30 分钟后检查 `log.txt` 有训练日志
- [ ] 5 小时后检查 55 epoch 中期结果

---

**现在可以直接复制命令到服务器启动实验了！**
