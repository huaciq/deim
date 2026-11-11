# RegionDense：36 轮训练及完整 test 评估结果

2026-11-11 完成。使用 val AP 最高的第 36 轮（index 35）best_stg2.pth；完整 test 11,613 张。
与 E000、E008 使用同一新服务器及相同评估流程：640 输入、EMA、原始官方 test 标注、top-300 后处理、COCO bbox maxDets=100。本次使用 GPU 4,5、双卡验证 batch 64。

## 整体指标

| 方法 | val AP | test AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E000 | 65.29 | 62.39 | 90.92 | 66.35 | 59.01 | 74.65 | 71.72 | 70.88 |
| E008 | 65.86 | 63.32 | 91.10 | 67.11 | 60.97 | 75.25 | 71.49 | 70.72 |
| RegionDense | 66.12 | 63.43 | 90.96 | 67.57 | 61.94 | 75.17 | 68.39 | 70.92 |
| 相对 E000 | +0.83 | +1.04 | +0.04 | +1.23 | +2.93 | +0.52 | -3.33 | +0.03 |
| 相对 E008 | +0.26 | +0.11 | -0.14 | +0.46 | +0.96 | -0.09 | -3.10 | +0.20 |

## 各类别 test AP

| 类别 | E000 | E008 | RegionDense | 相对 E008 | RegionDense AP75 |
|---|---:|---:|---:|---:|---:|
| ship | 72.90 | 73.41 | 73.80 | +0.39 | 83.53 |
| aircraft | 49.80 | 49.61 | 49.16 | -0.45 | 48.69 |
| car | 71.90 | 71.79 | 72.13 | +0.34 | 85.99 |
| tank | 53.91 | 56.61 | 56.29 | -0.32 | 50.65 |
| bridge | 44.87 | 46.98 | 47.31 | +0.33 | 43.98 |
| harbor | 80.95 | 81.50 | 81.87 | +0.38 | 92.61 |

APs 按 COCO 目标面积定义，不等于长边 <16 px；本轮未另跑该长边分组。

## 训练效率

| 方法 | 总训练时间 | 首次 val AP≥60 的轮号 | 首次 val AP≥65 的轮号 |
|---|---|---:|---:|
| E000 | 17:37:45 | 26 | 35 |
| E008 | 18:01:28 | 16 | 30 |
| RegionDense | 17:54:10 | 15 | 28 |

轮号从 1 开始。达到某 AP 的轮数较少不直接等于实际耗时更少；总训练时间基本相近。

## 判读

- 相对原 E000：val +0.83、test +1.04、test APs +2.93。
- 相对更强的 E008：val +0.26、test +0.11；AP75 +0.46、APs +0.96，但 APl -3.10。
- tank 总 AP -0.32，bridge +0.33，不能声称两个关键类别都得到提升。
- 未达到事先约定的 E008 +0.5 val AP（约 66.36）晋级线。当前是单种子、有局部收益但整体提升有限的候选，尚不能证明稳定整体增益。
- 本次 test 由用户明确要求执行，未据 test 选权重或修改方法参数。
- 这些汇总指标不足以把大目标退化具体归因于某一种裁图、上下文或采样机制。

## 可追溯结果

- 训练目录：`/root/zxy/experiments/regiondense_deim_hgnetv2_m_sardet100k_region_dense_36e_b32_seed0_20261110_203618_BjgqFF`
- test 目录：`/root/zxy/experiments/regiondense_deim_hgnetv2_m_sardet100k_region_dense_36e_b32_seed0_20261110_203618_BjgqFF/test_eval_best_stg2_LWGUQA`
- 训练及评估代码 revision：`6e305c3d725563382c977f2db087c81c968faaf8`。
- 原始 COCO precision/recall：同目录 eval.pth；精确数值：comparison.json、metrics.json；选择依据：selection.json；命令：command.sh；完整输出：console.log。

```bash
conda activate deim_zxy
cd /root/zxy/deim
export OUT=/root/zxy/experiments/regiondense_deim_hgnetv2_m_sardet100k_region_dense_36e_b32_seed0_20261110_203618_BjgqFF
export EVAL=/root/zxy/experiments/regiondense_deim_hgnetv2_m_sardet100k_region_dense_36e_b32_seed0_20261110_203618_BjgqFF/test_eval_best_stg2_LWGUQA
cat "$EVAL/metrics.json"
tail -n 25 "$EVAL/console.log"
```

独立复现 test 时使用新的目录，避免覆盖已完成的评估：

```bash
export REEVAL=$(mktemp -d "$OUT/test_eval_best_stg2_recheck_XXXXXX")
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/best_stg2.pth" \
  --test-only --seed 0 --use-amp --output-dir "$REEVAL" --summary-dir "$REEVAL/summary" \
  -u val_dataloader.dataset.img_folder=/root/zxy/TCRA-SAR/datasets/SARDet_100K/JPEGImages/test \
  val_dataloader.dataset.ann_file=/root/zxy/TCRA-SAR/datasets/SARDet_100K/Annotations/test.json \
  > "$REEVAL/console.log" 2>&1
```
