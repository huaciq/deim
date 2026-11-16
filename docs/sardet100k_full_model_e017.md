# E017：完整模型 = E013-v2 方向融合（主线）+ E016 RegionDenseSAR v3（辅助）

日期：2026-11-15 启动，2026-11-16 完成并评估 test。第一次把主线与辅助方法一起训练，检验两者是否叠加。结论：叠加成立，val 68.81、test 65.84，定为论文完整模型。

## 配置

`configs/deim_dfine/deim_hgnetv2_m_sardet100k_orientation_fusion_v2_region_dense_adaptive_36e_b32.yml` = E016 配置 + E013-v2 的
`DEIM.orientation_fusion`（projected、vflip、hidden 256、3 层、groups 8、reduction 4、init_bias 2.0、checkpoint_views）。
`tests/test_full_model_e017_config.py` 检查：相对 E016 只多出方向融合；方向融合各项与 E013-v2 相同；相对 E013-v2 只有训练数据构造不同
（RegionDenseSAR v3 计划代替 Mosaic/MixUp）。复用 E016 已生成并核对过的 v3 计划
`/root/zxy/experiments/region_banks/sardet100k_train_region_dense_sar_v3_seed0_b32`。

其余与 E000 36 轮协议一致：seed 0，全局 batch 32（GPU 4、5 各 16），COCO-M 初始化 `-t deim_dfine_hgnetv2_m_coco_90e.pth`，
AdamW 4e-4 / backbone 4e-5，flatcosine（warmup 2000、flat 20、no-aug 4），EMA 0.9999，AMP，多尺度 480–800 至 epoch 32，每轮完整 val。

## 判读

对照（同服务器、单种子）：E013-v2 val 67.26 / test 64.74，E016 val 66.33 / test 64.12，E000 val 65.29 / test 62.39。
- 组合 val 明显高于 E013-v2：互补成立，作为论文完整模型。
- 组合与 E013-v2 持平或更低：两者不叠加，需要先找原因（例如方向视角与区域拼图的相互作用），再定结构。
按类别重点看：tank、小目标（E016 的贡献）是否在组合中保留；bridge、harbor（E013-v2 的贡献）是否保留。

## 结果（2026-11-16）

36 轮完成，训练耗时 27:46:22（E000 的 1.58 倍）。按最高 val AP 选第 33 轮（index 32）的 `best_stg2.pth`（`last_epoch=32`），EMA 权重。
第 33–36 轮 val AP 为 68.81 / 68.80 / 68.75 / 68.81，已走平。test 为官方 11,613 张图，640 输入，GPU 4、5，全局评估 batch 64，top-300，
maxDets=100；训练与评估代码均为 `31b3eb0`，test 未参与选择。val 用同一权重重新评估一次以取得按类 AP，与训练日志一致（68.81）。

| 划分 | 方法 | AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| val | E000 | 65.29 | 93.15 | 70.26 | 60.18 | 76.63 | 72.09 | 73.36 |
| val | E013-v2 | 67.26 | 94.53 | 72.86 | 61.91 | 78.73 | 71.50 | 74.41 |
| val | E016 | 66.33 | 93.46 | 71.22 | 62.13 | 78.01 | 72.64 | 73.46 |
| val | **E017** | **68.81** | 95.25 | 73.91 | 64.24 | 80.39 | 73.82 | 75.35 |
| test | E000 | 62.39 | 90.92 | 66.35 | 59.01 | 74.65 | 71.72 | 70.88 |
| test | E013-v2 | 64.74 | 92.61 | 69.26 | 61.15 | 76.57 | 71.91 | 72.25 |
| test | E016 | 64.12 | 91.49 | 67.99 | 61.78 | 76.13 | 71.61 | 71.54 |
| test | **E017** | **65.84** | 92.73 | 70.30 | 63.25 | 77.50 | 72.71 | 73.16 |

| 类别（test AP） | E000 | E013-v2 | E016 | E017 | E017 − E000 | E017 − E013-v2 | E017 − E016 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ship | 72.90 | 72.74 | 73.86 | 73.86 | +0.96 | +1.12 | +0.00 |
| aircraft | 49.80 | 51.38 | 51.01 | 51.34 | +1.54 | −0.04 | +0.33 |
| car | 71.90 | 72.02 | 72.76 | 72.83 | +0.93 | +0.81 | +0.07 |
| tank | 53.91 | 55.05 | 56.70 | 57.50 | +3.59 | +2.45 | +0.80 |
| bridge | 44.87 | 51.80 | 47.32 | 52.99 | +8.12 | +1.19 | +5.67 |
| harbor | 80.95 | 85.44 | 83.07 | 86.55 | +5.60 | +1.11 | +3.48 |

val 按类相对 E000：ship +1.41、aircraft +0.64、car +0.33、tank +2.70、bridge +8.83、harbor +7.19。

判读（单种子）：

- **两者叠加，E017 定为论文完整模型。** 相对 E000，test +3.45、val +3.52；相对 E013-v2，test +1.10、val +1.55；相对 E016，test +1.72、val +2.48。
  单独增益之和为 test +4.08、val +3.01，组合在 test 上得到约 85%，在 val 上超过简单相加。
- **各自的贡献都保留了。** E016 带来的 tank 和小目标：test tank 57.50（高于 E016 的 56.70），APs 63.25（E000 +4.24、E013-v2 +2.10）。
  E013-v2 带来的 bridge、harbor：test 52.99 / 86.55，均高于 E013-v2 单独的 51.80 / 85.44。aircraft 与 E013-v2 持平。
- **增益比 E013-v2 单独时分布更均匀。** bridge 和 harbor 占 test 总增益的约 66%（E013-v2 单独时约 81%），六类全部为正。
- AP75 相对 E000 +3.95，大于 AP50 的 +1.81。APl 72.71，未出现 E013-v2 在 val 上的 APl 下降。

未解决：E013-v2 的翻转视图对照（E012、水平翻转视图、简单平均融合、测试时翻转）和计算量对照（DEIM-L）尚未运行，
方向互补的独立机制仍需这些实验支持。

## 结果文件与重算

评估目录：`$OUT/test_eval_selected_tH1Kds`（`eval.pth`、`metrics.json`、`comparison.json`、`selection.json`、`summarize_results.py`）；
val 重新评估：`$OUT/val_eval_selected_D9KIMj`。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
export OUT=/root/zxy/experiments/fullmodel_e017_deim_hgnetv2_m_sardet100k_orientation_fusion_v2_region_dense_adaptive_36e_b32_seed0_20261115_135344_kTy9zD
export EVAL="$OUT/test_eval_selected_tH1Kds" VEVAL="$OUT/val_eval_selected_D9KIMj"
cat "$EVAL/selection.json"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python "$EVAL/summarize_results.py"
```

独立重跑 test 时新建目录，先确认 GPU 4、5 空闲：

```bash
export REEVAL=$(mktemp -d "$OUT/test_eval_selected_recheck_XXXXXX")
ulimit -Sn 65535
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
