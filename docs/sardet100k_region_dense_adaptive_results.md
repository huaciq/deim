# E016 / RegionDenseSAR v3：36 轮训练与完整 test 结果

2026-11-14 记录（客户端日期）。E016 已完成 36 轮，训练耗时 17:29:55。按最高 val AP 选择第 36 轮（index 35）的 `best_stg2.pth`，checkpoint 内 `last_epoch=35`，使用 EMA 权重。

官方 test 共 11,613 张图，GPU 4、5 双卡，global eval batch 64，输入 640，原始官方标注、top-300 后处理、COCO bbox maxDets=100。训练和本次评估代码均为 `29978bc387b7d0e987a55d1e1f5914b7ba591e36`。选择权重时未使用 test。

## 总体结果

| 方法 | val AP | test AP | AP50 | AP75 | APs | APm | APl | AR100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E000 原 DEIM | 65.29 | 62.39 | 90.92 | 66.35 | 59.01 | 74.65 | 71.72 | 70.88 |
| E008 去掉 Dense O2O | 65.86 | 63.32 | 91.10 | 67.11 | 60.97 | 75.25 | 71.49 | 70.72 |
| E009 RegionDense v1 | 66.12 | 63.43 | 90.96 | 67.57 | 61.94 | 75.17 | 68.39 | 70.92 |
| E010 RegionDenseSAR v2 | 66.45 | 63.88 | 91.65 | 67.22 | 62.07 | 75.89 | 69.60 | 71.83 |
| E016 RegionDenseSAR v3 | 66.33 | 64.12 | 91.49 | 67.99 | 61.78 | 76.13 | 71.61 | 71.54 |
| v3 相对 E000 | +1.04 | +1.73 | +0.57 | +1.64 | +2.78 | +1.47 | −0.11 | +0.66 |
| v3 相对 E008 | +0.47 | +0.80 | +0.39 | +0.88 | +0.81 | +0.87 | +0.12 | +0.83 |
| v3 相对 E010 | −0.12 | +0.24 | −0.16 | +0.77 | −0.28 | +0.24 | +2.01 | −0.29 |

差值由未四舍五入结果计算，单位为 AP 百分点。APs/APm/APl 为 COCO 面积分组，不等同于长边 <16 px 的专门分组。

E016 的 val 细项为 AP50 93.46、AP75 71.22、APs 62.13、APm 78.01、APl 72.64、AR100 73.46。相对 E010，val APl +1.27、APs −0.76；test 也表现为大目标改善、小目标有所回落。

## 六类 test AP

| 类别 | E008 | E010 v2 | E016 v3 | v3 相对 E008 | v3 相对 v2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ship | 73.41 | 73.56 | 73.86 | +0.45 | +0.30 |
| aircraft | 49.61 | 50.55 | 51.01 | +1.40 | +0.46 |
| car | 71.79 | 72.14 | 72.76 | +0.97 | +0.62 |
| tank | 56.61 | 56.97 | 56.70 | +0.09 | −0.27 |
| bridge | 46.98 | 47.51 | 47.32 | +0.35 | −0.19 |
| harbor | 81.50 | 82.53 | 83.07 | +1.57 | +0.54 |

相对 E008，六类 test AP 均为正增益；相对 v2，ship、aircraft、car、harbor 上升，tank、bridge 下降。完整各类 AP50/AP75/APs/APm/APl 保存在评估目录的 `metrics.json`。

## 结果判读

非等分后备布局补回了 v2 的大目标损失：test APl 从 69.60 到 71.61，已接近 E008 的 71.49；val 大目标也提高。test AP75 同时提高 0.77，结果支持这项改造对部分目标的严格定位表现有帮助，但不能仅凭汇总指标证明具体因果机制。

小目标收益仍高于 E008（test APs +0.81），但比 v2 少 0.28。当前是监督分布发生取舍，不能写成所有尺度都优于 v2。

相对 E008，test AP +0.80；预先按 val 定义的 +0.8–1.0 目标未达到，val 实测为 +0.47。相对 v2 的总体 AP，val −0.12、test +0.24，两个划分方向不同。因此记录为“完成，目标未全部达到”，保留其大目标改善证据；不能依据 test 宣称已在整体上稳定超过 v2。当前均为单种子。

训练耗时 v2 17:33:40、v3 17:29:55；首次 val AP≥65 均在第 25 轮，v2 首次 ≥66 在第 31 轮，v3 在第 33 轮。未显示出收敛速度提升。E016 日志包含全部 16 个增强 epoch（index 4–19）。

## 原始记录与复现

训练目录：`/root/zxy/experiments/regiondense_sar_v3_deim_hgnetv2_m_sardet100k_region_dense_adaptive_36e_b32_seed0_20261114_114231_Hmbuun`。

本次 test 目录：训练目录下的 `test_eval_selected_CH4QSr/`，launcher PID `3263275` 已退出；结束后 GPU 4、5 均为 12 MiB / 0%。

评估目录保存 `selection.json`（val 选权重依据与两个 Git revision）、`command.sh` / `command.json`（实际命令）、`console.log`、`eval.pth`（完整 COCO precision/recall 对象）、`metrics.json`（精确指标）、`comparison.json`（五组精确结果及差值）、`summarize_results.py`（统计脚本）、`completion.json`（完成状态）。权重、原始结果和日志不提交 Git。

查看和重算已保存指标：

```bash
cd /root/zxy/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/root/zxy/experiments/regiondense_sar_v3_deim_hgnetv2_m_sardet100k_region_dense_adaptive_36e_b32_seed0_20261114_114231_Hmbuun
export EVAL="$OUT/test_eval_selected_CH4QSr"
cat "$EVAL/selection.json"
cat "$EVAL/metrics.json"
tail -n 35 "$EVAL/console.log"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python "$EVAL/summarize_results.py"
```

需要独立重跑 test 时使用新目录；先确认 GPU 4、5 空闲：

```bash
export REEVAL=$(mktemp -d "$OUT/test_eval_selected_recheck_XXXXXX")
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
nvidia-smi -i 4,5
ps -p "$(cat "$REEVAL/launcher.pid")" -o pid,etime,args
```

完整数据准备、训练、恢复与 TensorBoard 命令见 [E016 方法与训练协议](sardet100k_region_dense_adaptive_experiment.md)。本次仅运行 test 与整理结果，未修改实验代码或训练配置。
