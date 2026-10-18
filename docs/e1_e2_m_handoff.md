# E1/E2/M 实验交接

## 实验概览

| 编号 | 配置文件 | 改动 | 预计时间 |
|---|---|---|---|
| E1-NWD | `deim_hgnetv2_s_ogsod_120e_nwd.yml` | 匹配和损失加入 NWD | 120e ≈ 6h |
| E2-HWD | `deim_hgnetv2_s_ogsod_120e_hwd.yml` | Haar 下采样注入 P2 细节 | 120e ≈ 6h |
| M-baseline | `deim_hgnetv2_m_ogsod_120e.yml` | M 型号作为效率对比 | 120e ≈ 6h+ |

三个实验用同一套 GPU（两张 4090），需串行执行。

---

## 服务器更新与启动

### 0. 准备：下载 M 型号预训练权重

```bash
# 从 DEIM 官方 README 下载 M 的 COCO 权重到本地（约 19M 参数，52.7 AP）
# Google Drive 链接: https://drive.google.com/file/d/18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8/view?usp=drive_link
# 下载后放到服务器：
mkdir -p /home/zxy/sar/pretrain/deim
# 手动上传或用 gdown:
# pip install gdown
# gdown 18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8 -O /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
```

如果下载失败，M 基线可以先跳过，E1 和 E2 不依赖这个文件。

### 1. 更新代码

```bash
cd /home/zxy/sar/repos/deim
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
conda activate deim_zxy
```

确认拉取到最新 commit `7d7fef7`（"Add E1/E2 ablations..."）。

### 2. 启动 E1（NWD）

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_nwd/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_nwd.yml "$OUT/config.yml"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_nwd.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
disown
echo "E1-NWD started: $OUT"
```

### 3. 启动 E2（HWD）

**等 E1 跑完后**（约 6 小时），启动 E2：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_hwd/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_hwd.yml "$OUT/config.yml"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_hwd.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
disown
echo "E2-HWD started: $OUT"
```

### 4. 启动 M 基线

**等 E2 跑完后**，启动 M：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OUT=/home/zxy/sar/experiments/deim_dfine_m_ogsod_baseline_120e/seed0_$(date +%Y%m%d_%H%M%S)
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
git rev-parse HEAD > "$OUT/git_revision.txt"
cp configs/deim_dfine/deim_hgnetv2_m_ogsod_120e.yml "$OUT/config.yml"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c configs/deim_dfine/deim_hgnetv2_m_ogsod_120e.yml --seed 0 --use-amp \
  -t /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth \
  --output-dir "$OUT" > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
disown
echo "M-baseline started: $OUT"
```

如果 M 的 COCO 权重下载失败，这步会报错。可以先跑 E1、E2，M 的对比之后再补。

---

## 监控与恢复

### 查看运行状态

```bash
# 查看 GPU 占用
nvidia-smi

# 查看实时日志（替换为实际输出目录）
tail -f /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_nwd/seed0_YYYYMMDD_HHMMSS/console.log

# 查看训练进度（最后几行）
tail -n 5 /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_nwd/seed0_YYYYMMDD_HHMMSS/log.txt
```

### 恢复中断的训练

如果训练中断（例如连接断开、进程意外终止），用 `-r` 恢复：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
OLD=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_nwd/seed0_YYYYMMDD_HHMMSS  # 改成实际目录
OUT=/home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_nwd/seed0_resume_$(date +%Y%m%d_%H%M%S)
mkdir "$OUT"
cp -n "$OLD/best_stg1.pth" "$OUT/best_stg1.pth"
cp -n "$OLD/config.yml" "$OUT/original_config.yml"
git rev-parse HEAD > "$OUT/git_revision.txt"
printf '%s\n' "resume=$OLD/last.pth" "output_dir=$OUT" > "$OUT/resume_notes.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c configs/deim_dfine/deim_hgnetv2_s_ogsod_120e_nwd.yml --seed 0 --use-amp \
  -r "$OLD/last.pth" --output-dir "$OUT" \
  > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
disown
echo "Resumed: $OUT"
```

同理适用于 E2 和 M。

---

## 结果提取

### 查看最终指标

```bash
# 找到 log.txt 里最好的 epoch（best_stat 行）
grep best_stat /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_nwd/seed0_YYYYMMDD_HHMMSS/log.txt | tail -1

# 提取最佳 epoch 的 COCO 评测结果
# 假设最佳 epoch 是 119，对应 log.txt 里的那一行
tail -n 5 /home/zxy/sar/experiments/deim_dfine_s_ogsod_120e_nwd/seed0_YYYYMMDD_HHMMSS/log.txt | python3 -c "
import json, sys
for line in sys.stdin:
    d = json.loads(line)
    if 'test_coco_eval_bbox' in d:
        print(f\"Epoch {d['epoch']}: AP={d['test_coco_eval_bbox'][0]:.4f} AP50={d['test_coco_eval_bbox'][1]:.4f} AP75={d['test_coco_eval_bbox'][2]:.4f} APs={d['test_coco_eval_bbox'][3]:.4f}\")
"
```

### 对比表

把三个实验的最佳 epoch 结果整理成表：

| 模型 | AP | AP50 | AP75 | APs | APm | APl | 参数量 | 备注 |
|---|---|---|---|---|---|---|---|---|
| S-baseline-120e | 52.7 | 87.7 | 50.9 | 46.1 | 80.4 | 81.0 | 10M | 已完成 |
| S+NWD (E1) | ? | ? | ? | ? | ? | ? | 10M | 匹配+损失加 NWD |
| S+HWD (E2) | ? | ? | ? | ? | ? | ? | ~10.1M | P2 注入，略增参数 |
| M-baseline | ? | ? | ? | ? | ? | ? | 19M | 效率对比 |

**判断标准**：
- 如果 E1 或 E2 的 AP 比基线高 **0.5 以上**，视为有效。
- 重点看 **APs 和 AP75** 的涨幅，这两个指标直接对应小目标定位精度。
- 如果某个方向有效，后续再加 2 到 3 个 seed 跑置信区间。

---

## 预期结果与后续

### 如果 E1（NWD）有效
- 涨幅主要在 AP75 和 APs。
- 后续方向：把 NWD 也用到 MAL 的目标质量评分里（替换 IoU），作为 E1b。

### 如果 E2（HWD）有效
- 涨幅主要在 APs，油罐类别改善明显。
- 后续方向：在 P2 注入路径上加 SAR 噪声抑制模块（例如频域滤波）。

### 如果两者都有效
- E3 组合 E1 + E2，测试是否叠加。
- E4 加入形状感知模块（点状 vs 线状），作为原创贡献点。

### 如果都无效或涨幅 < 0.5
- 回到诊断，重新分析哪里出了问题（例如增强策略、学习率、NWD 常数）。
- 考虑换一个方向（例如 query 选择、损失权重调整）。

---

## 关于文献收集

暂时不需要。**等这三个实验跑完、知道哪个方向有效后，再针对性地找 2 到 3 篇论文深入改**。

到时候需要收集的类别：
1. **如果 NWD 有效**：找其他小目标检测的尺度不敏感损失（例如 RFLA、DotD、SimD），看能不能进一步改进。
2. **如果 HWD 有效**：找频域/小波在 SAR 去噪、细节保留上的应用（例如 FreqFusion、WTConv）。
3. **形状感知（E3 的候选）**：找条形卷积、方向卷积的论文（Strip Pooling、LSKNet、PKINet）。
4. **SAR 小目标检测 baseline**：找最近两年在 OGSOD、SSDD、HRSID 上的方法，作为 Related Work 和对比实验。

现在先不收集，避免被文献带偏。
