# 单卡 baseline 启动命令

## 配置说明

- **单卡训练**：`CUDA_VISIBLE_DEVICES=0`，`nproc_per_node=1`
- **Batch size**：32（全局 batch = 32，单卡 batch = 32）
- **增强策略**：移除了 RandomZoomOut 和 RandomIoUCrop
- **其他设置**：与双卡 baseline 一致（120 epoch, warmup 1000, flat 60, no_aug 8）

## 服务器启动

### 1. 更新代码

```bash
cd /home/zxy/sar/repos/deim
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
conda activate deim_zxy
```

确认拉取到 commit `8bb3f68`（"Add single-GPU baseline..."）。

### 2. 启动训练

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_noaug/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_noaug.yml "$OUT/config.yml"
nohup env CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_1gpu_noaug.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
disown
echo "Single-GPU baseline started: $OUT"
echo "PID: $PID"
```

### 3. 监控

```bash
# 查看 GPU 占用（应该只有 GPU 0 在用）
nvidia-smi

# 查看实时日志
tail -f /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_noaug/seed0_*/console.log

# 查看训练进度
tail -n 3 /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_noaug/seed0_*/log.txt
```

### 4. 检查 55 epoch 中期指标

```bash
RUN=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_1gpu_noaug/seed0_*  # 替换为实际目录
sed -n '56p' "$RUN/log.txt" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
print(f\"1GPU-noaug@55e: AP={d['test_coco_eval_bbox'][0]*100:.1f} APs={d['test_coco_eval_bbox'][3]*100:.1f}\")
"
```

**对比参考**：
- 双卡 baseline@55e: AP 30.0, APs 22.2
- 如果单卡@55e 在 29~31 之间，说明单卡影响可控
- 如果 > 31，说明移除增强有效

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

## 预期时间

- **单卡每 epoch**：约 5~6 分钟（双卡是 3 分钟）
- **120 epoch 总时间**：约 10~12 小时

---

## 对比表（占位）

| 模型 | GPU | Batch | 增强 | AP@120e | APs | AP75 | 备注 |
|---|---|---|---|---|---|---|---|
| S-baseline | 2×4090 | 32 (16/卡) | 全部 | 52.7 | 46.1 | 50.9 | 原双卡基线 |
| S-1GPU-noaug | 1×4090 | 32 (32/卡) | 移除2个 | ? | ? | ? | **新单卡基线** |

把结果填进这里，重点看：
- 如果单卡 AP 在 52~54 之间，说明单卡影响 < 1 个点，可以继续用单卡
- 如果 > 54，说明移除增强带来 1.5+ 个点提升，那就值得

---

## 后续实验

如果这个单卡 baseline 的 AP 可接受（比如 > 52），后续所有实验（F2 reg_scale、F3 lowlr、seed 补充）都用单卡、相同 batch 32、移除增强的协议。

如果 AP 掉到 50 以下，说明单卡 batch 32 不稳定，需要考虑梯度累积或换回双卡。
