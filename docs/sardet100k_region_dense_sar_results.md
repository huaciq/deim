# RegionDenseSAR v2：36 轮训练及完整 test 评估结果

2026-11-12 记录。实验 E010，按最高 val AP 选择第 36 轮（index 35）的 `best_stg2.pth`，checkpoint 内的 last_epoch 也是 35，使用 EMA 权重。完整 test 为官方 11,613 张图。

与 E000、E008、E009 使用同一新服务器及相同评估流程：640 输入、原始官方 test 标注、top-300 后处理、COCO bbox maxDets=100。本次使用 GPU 4、5，双卡验证 global batch 64。未使用 test 选择权重或参数。

## 整体指标

| 方法 | val AP | test AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E000 原 DEIM | 65.29 | 62.39 | 90.92 | 66.35 | 59.01 | 74.65 | 71.72 | 70.88 |
| E008 去掉 Dense O2O | 65.86 | 63.32 | 91.10 | 67.11 | 60.97 | 75.25 | 71.49 | 70.72 |
| E009 RegionDense v1 | 66.12 | 63.43 | 90.96 | 67.57 | 61.94 | 75.17 | 68.39 | 70.92 |
| E010 RegionDenseSAR v2 | 66.45 | 63.88 | 91.65 | 67.22 | 62.07 | 75.89 | 69.60 | 71.83 |
| v2 相对 E000 | +1.16 | +1.49 | +0.73 | +0.87 | +3.06 | +1.23 | -2.12 | +0.95 |
| v2 相对 E008 | +0.59 | +0.56 | +0.54 | +0.10 | +1.09 | +0.63 | -1.89 | +1.12 |
| v2 相对 v1 | +0.33 | +0.45 | +0.69 | -0.36 | +0.13 | +0.72 | +1.21 | +0.92 |

差值由未四舍五入数值计算，单位是 AP 百分点。APs/APm/APl 使用 COCO 面积分组，不能替代长边 <16 px 的专门分组；本次没有另外运行该长边分组。

## 六类 test AP

| 类别 | E000 | E008 | v1 | v2 | v2 相对 E008 | v2 相对 v1 |
|---|---:|---:|---:|---:|---:|---:|
| ship | 72.90 | 73.41 | 73.80 | 73.56 | +0.15 | -0.24 |
| aircraft | 49.80 | 49.61 | 49.16 | 50.55 | +0.94 | +1.39 |
| car | 71.90 | 71.79 | 72.13 | 72.14 | +0.35 | +0.01 |
| tank | 53.91 | 56.61 | 56.29 | 56.97 | +0.36 | +0.68 |
| bridge | 44.87 | 46.98 | 47.31 | 47.51 | +0.54 | +0.20 |
| harbor | 80.95 | 81.50 | 81.87 | 82.53 | +1.03 | +0.65 |

相对 E008，六类 test AP 均上涨；相对 v1，ship 略降。完整各类 AP50/AP75/APs/APm/APl 在评估目录 metrics.json 和 comparison.json 中。

## 训练过程

| 方法 | 总训练时间 | 首次 val AP≥60 的轮号 | 首次 val AP≥65 的轮号 | 首次 val AP≥66 的轮号 |
|---|---|---:|---:|---:|
| E000 | 17:37:45 | 26 | 35 | 未达到 |
| E008 | 18:01:28 | 16 | 30 | 未达到 |
| E009 v1 | 17:54:10 | 15 | 28 | 36 |
| E010 v2 | 17:33:40 | 15 | 25 | 31 |

轮号从 1 开始。v2 训练日志包含全部 16 个增强 epoch（index 4–19）的固定组图计划记录。总训练时长和 E000 相近；达到 val AP 65 的轮号从 E008 的 30 提前到 25。达到阈值的轮数与实际墙钟时间需区分；这里仍是相同 36 轮预算比较，未证明充分收敛。

## 当前结论

- v2 相对直接对照 E008：val AP +0.59、test AP +0.56，达到原先约定的 E008 +0.5 val AP 筛选线；支持保留为辅助模块候选。
- 相对 v1，总体 test AP +0.45，APs +0.13，APl +1.21：小目标收益基本保留，大目标退化有所减轻。
- 大目标问题尚未解决：test APl 仍比 E008 低 1.89；test AP75 比 v1 低 0.36。不能把这版写成各项指标都改善。
- 本轮同时改了源图采样和背景选择，结果支持完整 v2 方案；两部分各自的 AP 贡献尚未单独分开。
- 当前为单种子。后续作为论文辅助创新时，应在确定的主要方法上验证组合贡献及稳定性。

## 原始结果与复现

训练目录：`/root/zxy/experiments/regiondense_sar_v2_deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32_seed0_20261111_214950_thYezk`。

test 目录：上述训练目录下 `test_eval_best_stg2_lQ2o9V/`。评估 launcher PID `3582739` 已退出，GPU 4、5 评估结束后均回到 12 MiB / 0% 利用率。

训练与 test 评估使用代码 `33172bb9d9f1ceafedcf28d71f58a743d3d46674`。完整训练协议和启动/恢复命令见 [正式实验记录](sardet100k_region_dense_sar_experiment.md)。

关键文件：

- `selection.json`：val 选择依据、权重路径、checkpoint epoch、训练与评估 revision。
- `command.sh` / `command.json`：实际 test 命令。
- `console.log`：完整 test 控制台输出。
- `eval.pth`：COCO precision/recall 原始评估对象，包含完整 11,613 张图。
- `metrics.json`：E010 精确指标；`comparison.json`：四组精确指标及差值。
- `summarize_results.py`：从原始 COCO 数组计算上述指标的脚本。

查看本次结果：

```bash
conda activate deim_zxy
cd /root/zxy/deim
export OUT=/root/zxy/experiments/regiondense_sar_v2_deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32_seed0_20261111_214950_thYezk
export EVAL="$OUT/test_eval_best_stg2_lQ2o9V"
cat "$EVAL/metrics.json"
tail -n 25 "$EVAL/console.log"
```

需要独立重跑 test 时，用新目录保存，避免覆盖本次结果：

```bash
export REEVAL=$(mktemp -d "$OUT/test_eval_best_stg2_recheck_XXXXXX")
nohup env CUDA_VISIBLE_DEVICES=4,5 PYTHONUNBUFFERED=1 \
torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/best_stg2.pth" \
  --test-only --seed 0 --use-amp --output-dir "$REEVAL" --summary-dir "$REEVAL/summary" \
  -u val_dataloader.dataset.img_folder=/root/zxy/TCRA-SAR/datasets/SARDet_100K/JPEGImages/test \
     val_dataloader.dataset.ann_file=/root/zxy/TCRA-SAR/datasets/SARDet_100K/Annotations/test.json \
  > "$REEVAL/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$REEVAL/launcher.pid"
disown "$PID"
tail -f "$REEVAL/console.log"
```
