# E021：完整模型的 ImageNet 初始化版本（与 E019 配对）

日期：2026-11-19。

## 要回答的问题

E019 显示，在 36 轮协议下 COCO 检测初始化只比 ImageNet 骨干初始化高约 1 AP（val 1.18，test 1.02）。
剩下的问题是：完整模型（E017 = OCEF / E013-v2 + SCRM / E016）相对基线的提升，是否依赖 COCO 检测权重。

| 初始化 | 基线 | 完整模型 | 提升 |
|---|---|---|---|
| COCO 检测权重（`-t`） | E000：val 65.29 / test 62.39 | E017：val 68.81 / test 65.84 | val +3.52 / test +3.45 |
| ImageNet 骨干 | E019：val 64.11 / test 61.37 | **E021**：val 68.34 / test 65.31（旧服务器） | val +4.23 / test +3.94 |

结论（2026-11-21）：q = 1.20（val）/ 1.14（test），≥ 0.7，**方法增益不依赖 COCO 检测初始化**。详见文末"结果"。

## 配置与实现

- 配置：`configs/deim_dfine/deim_hgnetv2_m_sardet100k_imagenet_init_full_model_36e_b32.yml`，包含 E017 配置，只改
  `HGNetv2.pretrained: True` 和 `HGNetv2.local_model_dir: /root/zxy/pretrain/hgnetv2/`（与 E019 相同）。
  用 `INITIAL_WEIGHTS=none` 启动，不传 `-t`；encoder、decoder 和方向融合门控从头初始化。
- 复用 E016/E017 的 v3 拼图计划 `/root/zxy/experiments/region_banks/sardet100k_train_region_dense_sar_v3_seed0_b32`。
- 定向测试 `tests/test_imagenet_init_full_model_e021.py`：E021 相对 E017 只有上述两项不同且取值与 E019 相同；
  E021 相对 E019 改动的配置项，正好等于 E017 相对 E000 改动的配置项，取值也相同。本地与 E017、E019 的配置测试共 8 项通过。

## 放在哪台服务器

原计划放在 8 卡服务器 GPU 4、5（与 E000、E017、E019 同服务器）。2026-11-19 用户决定：GPU 4、5 另有用途，
**E021 改在旧服务器 GPU 0、1 运行**，在旧服务器上重建 v2、v3 拼图计划，不补跑 ImageNet 基线。
因此 E021 − E019 是跨服务器的差值；两台服务器同一基线相差约 0.1–0.4 AP（E000 / E000-old），判读时把这一量级考虑在内。
旧服务器的配置、计划重建和命令见文末"旧服务器版本"。

## 协议

与 E017 相同：SARDet-100K 官方 train / val，seed 0，全局 batch 32（GPU 4、5 各 16），36 轮（flat 20，末 4 轮无增强），
AdamW 主 LR 4e-4 / 骨干 4e-5，EMA 0.9999，AMP，RegionDenseSAR v3 代替 Mosaic / MixUp，训练多尺度 480–800、评估 640，每轮完整 val。
唯一区别是初始化。按最高 val AP 选权重，test 训练结束后评估一次，不参与选择。预计约 28 小时（E017 为 27:46:22）。

## 事先定好的判读

记 q = (E021 − E019) / (E017 − E000)，以 val 为主（分母 3.52），test 同样计算（分母 3.45）：

| 结果 | 判读 |
|---|---|
| q ≥ 0.7（val 提升 ≥ 2.46） | 方法增益基本不依赖 COCO 检测初始化 |
| 0.3 < q < 0.7 | 一部分增益依赖 COCO 初始化，两种初始化都报告 |
| q ≤ 0.3（val 提升 ≤ 1.06） | 增益主要依赖 COCO 初始化，论文须以 COCO 初始化为前提表述 |

按类看 OCEF 带来的 bridge / harbor 和 SCRM 带来的 tank / 小目标是否保留。单次运行，按上表直接读。
不为 ImageNet 初始化另调轮数；E019 末轮仍在上升，E021 同样报告逐轮曲线。

## 服务器命令（8 卡服务器）

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
git status --short --branch
git -c pull.rebase=false pull --ff-only origin main
git rev-parse --short HEAD
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_imagenet_init_full_model_e021 -v
GPU_IDS=4,5 EXPERIMENT_ROOT=/root/zxy/experiments INITIAL_WEIGHTS=none RUN_PREFIX=imagenet_full_e021 \
  bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sardet100k_imagenet_init_full_model_36e_b32.yml
```

服务器连不上 GitHub 时，在本地生成 bundle 并传过去：

```bash
git bundle create deim_main.bundle <服务器当前提交>..main
scp -P 20106 deim_main.bundle root@10.11.154.192:/root/zxy/deim_main.bundle
```

```bash
cd /root/zxy/deim && git pull --ff-only /root/zxy/deim_main.bundle main
```

监控与恢复（先设置 `OUT` 为 `STARTED` 行中的目录）：

```bash
tail -f "$OUT/console.log"
```

```bash
nvidia-smi -i 4,5; ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
```

```bash
python -c 'import json,os; [print(r["epoch"]+1, [round(100*x,2) for x in r["test_coco_eval_bbox"][:6]]) for r in map(json.loads, filter(str.strip, open(os.environ["OUT"]+"/log.txt")))]'
```

```bash
GPU_IDS=4,5 bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

训练结束后按 val 选权重并评估 test，命令与 [E017](sardet100k_full_model_e017.md) 相同；对比可用
`tools/sardet100k/compare_runs.py`（`--run E019=... --run E021=...`）。

## 旧服务器版本（实际运行）

### 配置

| 文件 | 用途 | 相对 8 卡服务器版本的差别 |
|---|---|---|
| `configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32_old_server.yml` | 重建 v2 计划（E010 的计划，v3 的基础） | 只改 train/val 数据路径、`plan_dir`、输出目录 |
| `configs/deim_dfine/deim_hgnetv2_m_sardet100k_imagenet_init_full_model_36e_b32_old_server.yml` | 重建 v3 计划，并训练 E021 | 只改 train/val 数据路径、`plan_dir`、`HGNetv2.local_model_dir: /home/zxy/sar/pretrain/hgnetv2/`、输出目录 |

数据：`/data2/zxy/sar/datasets/SARDet_100K`（与 E000-old 相同）。计划目录：
`/data2/zxy/sar/experiments/region_banks/sardet100k_train_region_dense_sar_{v2,v3}_seed0_b32`（v3 约 4.7 GB）。

定向测试 `tests/test_e021_old_server.py`：两份配置相对 8 卡服务器版本只有路径不同；数据路径与 E000-old 相同；
v2 计划设置等于 v3 设置去掉 `adaptive_partition`（v3 构建脚本要求这一点）；`tools/sardet100k/plan_summary.py` 读数正确。
本地与 E021 配置测试共 7 项通过。

### 计划重建与核对

两套计划都只读 train 标注和图像，按 seed 0 生成，与 8 卡服务器使用同一代码。8 卡服务器上 v3 构建用时约 9 分钟（特征 432 s）。
构建完成后用 `tools/sardet100k/plan_summary.py` 打印统计，与 8 卡服务器的参考值逐项对照：

| 项目 | v2 参考值 | v3 参考值 |
|---|---|---|
| source_images / canvases_per_epoch / steps_per_epoch | 94493 / 94464 / 2952 | 94493 / 94464 / 2952 |
| completed_source_passes / every_pass_complete | 23 / true | 26 / true |
| remaining_pass_sources_used | 64142 | 10599 |
| total_source_exposures | 2237481 | 2467417 |
| instance_exposure | [23, 24] | [26, 27] |
| total_class_counts | [2210272, 964111, 226676, 571376, 653754, 78278] | [2437545, 1062059, 249770, 630799, 721017, 86359] |
| feature_selection_fallbacks | 0 | 0（adaptive 也为 0） |
| first_epoch_routes | {"-1": 70501, "0": 10732, "1": 6466, "2": 6765} | {"-1": 52005, "0": 8664, "1": 6263, "2": 6448, "3": 10825, "4": 10259} |
| first_epoch_gt / last_epoch_gt | 293489 / 294782 | 324647 / 324744 |
| feature_valid_regions | 148356 | — |

读法：前两行与 settings 只由标注和设置决定，必须一致。其余项取决于背景特征，图像解码完全相同时也应一致；
若只有这些项略有差别，说明两台机器的 JPEG 解码有细微差异，计划仍是同一算法生成的有效 v3 计划，照常训练并在结果中注明。

**核对结果（2026-11-19，旧服务器，ce6d8a0）：** v2、v3 两套计划的上表各项与参考值完全一致，包括依赖背景特征的
类别总数、路线分布、feature_valid_regions 和 0 次回退。另外旧服务器的 last_epoch_routes 为
v2 {"-1": 70618, "0": 10742, "1": 6342, "2": 6762}、v3 {"-1": 51993, "0": 8723, "1": 6269, "2": 6525, "3": 10818, "4": 10136}，留作以后对照。
因此 E021-old 与 E017 使用的拼图计划可视为相同，E021 与 E017 之间只差初始化（服务器不同仍需注明）。

### 旧服务器命令

更新代码并检查：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
git status --short --branch
git fetch origin
git switch main
git -c pull.rebase=false pull --ff-only origin main
git log --oneline -1
python -m unittest tests.test_e021_old_server tests.test_imagenet_init_full_model_e021 -v
ls -l /home/zxy/sar/pretrain/hgnetv2/PPHGNetV2_B2_stage1.pth
df -h /data2
```

重建 v2、v3 计划（只用 CPU，后台顺序执行；计划目录已存在时脚本报错退出，不会覆盖）：

```bash
mkdir -p /data2/zxy/sar/experiments/diagnostics
export PREP=$(mktemp -d /data2/zxy/sar/experiments/diagnostics/region_dense_plans_old_server_XXXXXX)
nohup env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 bash -c '
python tools/build_region_dense_sar.py --config configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32_old_server.yml --workers 8 --world-size 2 &&
python tools/build_region_dense_adaptive.py --config configs/deim_dfine/deim_hgnetv2_m_sardet100k_imagenet_init_full_model_36e_b32_old_server.yml --base-plan-dir /data2/zxy/sar/experiments/region_banks/sardet100k_train_region_dense_sar_v2_seed0_b32 --workers 8 --world-size 2
' > "$PREP/build.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PREP/build.pid"
disown "$PID"
tail -f "$PREP/build.log"
```

日志最后一行是 `Prepared /data2/zxy/sar/experiments/region_banks/sardet100k_train_region_dense_sar_v3_seed0_b32; total elapsed ...` 即完成。打印两套计划的统计：

```bash
python tools/sardet100k/plan_summary.py --plan-dir /data2/zxy/sar/experiments/region_banks/sardet100k_train_region_dense_sar_v2_seed0_b32
python tools/sardet100k/plan_summary.py --plan-dir /data2/zxy/sar/experiments/region_banks/sardet100k_train_region_dense_sar_v3_seed0_b32
```

启动 E021（不加 `-t`；脚本只在 GPU 0、1 空闲时启动，nohup 后保存 PID 并 disown）：

```bash
GPU_IDS=0,1 EXPERIMENT_ROOT=/data2/zxy/sar/experiments INITIAL_WEIGHTS=none RUN_PREFIX=deim_m_sardet100k \
  bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sardet100k_imagenet_init_full_model_36e_b32_old_server.yml
export OUT=/data2/zxy/sar/experiments/<STARTED 行里的目录>
grep -c "Loaded stage1 B2 HGNetV2 from local file" "$OUT/console.log"
tail -1 "$OUT/start_command.sh"
```

`start_command.sh` 里不应有 `-t`。监控（重连后先 `cd /home/zxy/sar/repos/deim && conda activate deim_zxy` 并重新设置 OUT）：

```bash
tail -f "$OUT/console.log"
```

```bash
nvidia-smi -i 0,1; ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
```

```bash
python -c 'import json,os; [print(r["epoch"]+1, [round(100*x,2) for x in r["test_coco_eval_bbox"][:6]]) for r in map(json.loads, filter(str.strip, open(os.environ["OUT"]+"/log.txt")))]'
```

```bash
tensorboard --logdir "$OUT/summary" --host 127.0.0.1 --port 6006
```

中断后恢复（`-r last.pth`，不与 `-t` 同时使用）：

```bash
GPU_IDS=0,1 bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

训练结束后按 val 选权重、评估 test，并打印总体与按类 AP：

```bash
export CKPT=$(python -c 'import json,os; rows=[json.loads(l) for l in open(os.environ["OUT"]+"/log.txt") if l.strip()]; b=max(rows,key=lambda r:r["test_coco_eval_bbox"][0]); print("epochs=%d best_val_AP=%.2f best_epoch=%d" % (len(rows),100*b["test_coco_eval_bbox"][0],b["epoch"]+1), file=__import__("sys").stderr); print(os.environ["OUT"]+"/"+("best_stg2.pth" if b["epoch"]>=32 else "best_stg1.pth"))') && echo "$CKPT"
export EVAL=$(mktemp -d "$OUT/test_eval_selected_XXXXXX") && ulimit -Sn 65535
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py -c "$OUT/resolved_config.yml" -r "$CKPT" \
  --test-only --seed 0 --use-amp --output-dir "$EVAL" --summary-dir "$EVAL/summary" \
  -u val_dataloader.dataset.img_folder=/data2/zxy/sar/datasets/SARDet_100K/JPEGImages/test \
     val_dataloader.dataset.ann_file=/data2/zxy/sar/datasets/SARDet_100K/Annotations/test.json \
  2>&1 | tee "$EVAL/console.log"
python tools/sardet100k/compare_runs.py --test-ann /data2/zxy/sar/datasets/SARDet_100K/Annotations/test.json \
  --run E021-old="$OUT:$(basename "$EVAL")" --output "$EVAL/comparison.json"
```

E019 在 8 卡服务器上，E021 − E019 由结果记录时计算并注明跨服务器。

## 结果（2026-11-21，旧服务器 GPU 0、1，单次运行）

训练 36 轮，用时 1 天 1:00:33（25:00:33；E017 在 8 卡服务器为 27:46:22）。val 最佳为第 36 轮（末轮，`best_stg2.pth`），
test 只在训练结束后评估一次，不参与选择。v2/v3 拼图计划与 8 卡服务器逐项一致（见上文"核对结果"）。

| 划分 | 运行 | AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| val | E000：COCO 初始化基线 | 65.29 | 93.15 | 70.26 | 60.18 | 76.63 | 72.09 | 73.36 |
| val | E017：COCO 初始化完整模型 | 68.81 | 95.25 | 73.91 | 64.24 | 80.39 | 73.82 | 75.35 |
| val | E019：ImageNet 初始化基线 | 64.11 | 92.41 | 69.14 | 58.78 | 75.59 | 69.12 | 72.58 |
| val | **E021：ImageNet 初始化完整模型（旧服务器）** | 68.34 | 94.50 | 73.48 | 64.30 | 79.79 | 72.51 | 75.94 |
| test | E000 | 62.39 | 90.92 | 66.35 | 59.01 | 74.65 | 71.72 | 70.88 |
| test | E017 | 65.84 | 92.73 | 70.30 | 63.25 | 77.50 | 72.71 | 73.16 |
| test | E019 | 61.37 | 89.64 | 65.58 | 58.50 | 73.75 | 67.93 | 70.22 |
| test | **E021** | 65.31 | 92.17 | 69.56 | 63.73 | 76.80 | 72.25 | 73.44 |

方法增益在两种初始化下的对比（E021 − E019 跨服务器）：

| 划分 | 增益 | AP | APs | APl | ship | aircraft | car | tank | bridge | harbor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | COCO：E017 − E000 | +3.52 | +4.06 | +1.73 | +1.40 | +0.64 | +0.33 | +2.71 | +8.82 | +7.19 |
| val | ImageNet：E021 − E019 | +4.23 | +5.52 | +3.39 | +1.64 | +0.13 | +0.19 | +3.92 | +9.50 | +10.00 |
| test | COCO：E017 − E000 | +3.45 | +4.24 | +0.99 | +0.96 | +1.54 | +0.93 | +3.59 | +8.12 | +5.60 |
| test | ImageNet：E021 − E019 | +3.94 | +5.23 | +4.32 | +1.84 | +0.30 | −0.62 | +4.20 | +10.10 | +7.78 |

E021 按类 AP：val ship 74.69、aircraft 61.99、car 71.21、tank 60.46、bridge 54.56、harbor 87.11；
test 73.77、48.92、71.10、57.81、53.09、87.16。同服务器对照 E000-old（compare_runs）：val +3.19，test +2.61。

逐轮 val AP：第 1 轮 29.72，第 10 轮 53.49，第 20 轮 64.89，第 30 轮 67.66，第 33–36 轮 67.97、68.08、68.26、68.34，末轮仍在上升。

判读（按事先定好的标准，单次运行）：

- **q = 4.23 / 3.52 = 1.20（val），3.94 / 3.45 = 1.14（test），远高于 0.7：方法增益不依赖 COCO 检测初始化。**
  用 E000-old − E000 的服务器差（val −0.14，test +0.31）校正后 q 为 1.24 / 1.05；即使把 0.4 AP 的跨服务器差全部算作不利，
  val q 仍约 1.09。q 高于 1 的那部分（约 0.5–0.7 AP）在跨服务器和单次运行的波动范围内，不作为"ImageNet 下增益更大"的结论。
- **两个方法各自的特征都保留了。** OCEF 的 bridge / harbor：val +9.50 / +10.00，test +10.10 / +7.78；
  SCRM 的 tank / 小目标：tank val +3.92 / test +4.20，APs val +5.52 / test +5.23。均不小于 COCO 初始化下的对应增益。
- **完整模型上 COCO 初始化的贡献变小。** E021 比 E017 低 0.47（val）/ 0.53（test），基线上是 1.18 / 1.02；
  E019 最大的短板 APl（val −2.97 / test −3.79）在完整模型上缩小到 −1.31 / −0.46（含跨服务器差）。
- **不用 COCO 权重的完整模型仍高于用 COCO 权重的基线：** E021 − E000 = val +3.05 / test +2.92。
- aircraft 在 ImageNet 初始化下几乎没有增益（val +0.13 / test +0.30），但同一基线在两台服务器上 aircraft test 就差 1.70，
  这一项不单独解读。
- 和 E019 一样，末轮仍在上升；36 轮协议对 ImageNet 初始化略偏保守，按协议不另调轮数。

论文用途：初始化对照表的四格（E000 / E017 / E019 / E021）已齐。表中注明 E021 在旧服务器运行、计划逐项一致；
可以写明"方法在 COCO 检测初始化和 ImageNet 骨干初始化下分别带来 +3.45 / +3.94 test AP"。

结果文件：旧服务器运行目录
`/data2/zxy/sar/experiments/deim_m_sardet100k_deim_hgnetv2_m_sardet100k_imagenet_init_full_model_36e_b32_old_server_seed0_*`，
test 评估目录 `test_eval_selected_*/`（含 `comparison.json`，与 E000-old 的 `test_eval_best_stg2_R7nI3O/` 对照）。
