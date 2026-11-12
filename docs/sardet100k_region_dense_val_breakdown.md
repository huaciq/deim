# RegionDense 的 val 拆解与 E011（RegionDense + MixUp）

日期：2026-11-11。接 [RegionDense 设计](sardet100k_region_dense_design.md) 与 [RegionDense 结果](sardet100k_region_dense_results.md)。

要回答的问题：RegionDense（E009）的“大目标下降”到底落在哪里，下一步该怎么改。只用 val；test 的 APl 不作为调方法的依据。

## 口径

- E000（原 DEIM）、E008（去掉 Mosaic 与 MixUp）、E009（RegionDense）三次运行，都是 8 卡服务器、36 轮、b32、seed 0，各取 `best_stg2`（第 35 轮）。
- E008、E009 在 val 上重新推理（GPU 4、5，`e1_report.py collect`），E000 用已有的 `val_predictions.npz`。重算的官方指标与训练日志一致：65.29 / 65.86 / 66.11。
- 用 `tools/sardet100k/source_class_eval.py` 按来源 × 类别 × 尺寸拆开。尺寸按框的长边，分原图像素和 640 输入像素两种；工作点为 score ≥ .3、IoU ≥ .5。
- 服务器输出 `independent_verify_20261106/d11_region_dense_val/`（git `6e305c3`），本地 `analysis_outputs/d11_region_dense_val/`。单种子。

## 结果

### 按输入像素的长边（六类合并，AP：E000 / E008 / E009）

| 输入长边 | GT | AP | 漏检数 |
|---|---:|---|---|
| < 16 | 1,003 | 36.6 / 37.3 / 39.4 | 230 / 233 / 232 |
| 16–32 | 2,899 | 50.9 / 53.7 / 54.7 | 381 / 326 / 313 |
| 32–64 | 5,892 | 67.8 / 69.2 / 69.9 | 280 / 266 / 263 |
| 64–128 | 9,695 | 73.8 / 74.0 / 74.8 | 295 / **441 / 424** |
| ≥ 128 | 3,214 | 73.2 / 73.7 / 73.2 | 185 / 189 / 203 |

### 64 像素以上按类别（AP：E000 / E008 / E009）

| 类别 | 输入长边 | GT | AP | 漏检数 |
|---|---|---:|---|---|
| ship | 64–128 | 4,297 | 74.2 / 75.5 / 76.2 | 96 / 111 / 113 |
| ship | ≥ 128 | 1,644 | 78.1 / 79.2 / 79.4 | 30 / 46 / 41 |
| bridge | 64–128 | 617 | 60.1 / 63.2 / 63.7 | 53 / 42 / 35 |
| bridge | ≥ 128 | 350 | 71.1 / 74.0 / 72.4 | 19 / 12 / 13 |
| tank | 64–128 | 388 | 96.5 / 98.2 / 97.7 | 1 / 2 / 3 |
| harbor | ≥ 128 | 281 | 86.3 / 86.8 / 86.2 | 6 / 7 / 4 |
| **aircraft** | 64–128 | 3,839 | **63.7 / 61.6 / 62.5** | **135 / 277 / 264** |
| **aircraft** | ≥ 128 | 936 | **57.5 / 54.7 / 54.8** | 130 / 124 / 145 |

### 主要的来源 × 类别单元格（AP：E000 / E008 / E009）

| 单元格 | GT | AP | AP75 | 漏检数 |
|---|---:|---|---|---|
| OGSOD tank | 1,216 | 30.6 / 32.2 / 32.2 | 17.3 / 18.0 / 20.2 | 246 / 225 / 207 |
| OGSOD bridge | 3,224 | 45.0 / 47.8 / 47.8 | 41.0 / 44.2 / 44.5 | 450 / 352 / 346 |
| MSAR ship | 2,017 | 85.4 / 88.1 / 89.2 | 91.9 / 93.0 / 92.9 | 23 / 22 / 22 |
| MSAR aircraft | 115 | 74.2 / 82.5 / 87.2 | 86.2 / 92.6 / 93.9 | 4 / 4 / 4 |
| other_224 aircraft | 448 | 66.6 / 73.8 / 74.1 | 77.6 / 82.7 / 81.6 | 11 / 4 / 6 |
| other_256 ship | 5,080 | 69.3 / 70.2 / 70.8 | 82.9 / 84.3 / 85.1 | 129 / 156 / 158 |
| other_800 ship | 2,975 | 73.6 / 72.9 / 73.4 | 84.1 / 83.7 / 84.4 | 163 / 187 / 187 |
| **other_512 aircraft** | 4,631 | **62.6 / 59.8 / 60.9** | **63.2 / 57.5 / 59.4** | **261 / 399 / 406** |

other_512 aircraft（输入长边 64–128）的漏检构成：完全没有 IoU ≥ .5 的框 87 / 180 / 150，有框但分数低 30 / 67 / 105，定位不足 16 / 29 / 8。

## 判读

1. **“大目标下降”不是大目标的普遍现象，而是 512 像素来源的 aircraft 这一格。** 同样是 64 像素以上，ship、bridge、tank 在去掉 Dense O2O 之后都持平或上升；只有 aircraft 下降（64–128：−2.1，≥ 128：−2.8），而且从 E008 就开始了，不是 RegionDense 带来的。
2. **它是检出问题，不是定位问题。** other_512 aircraft 的漏检从 261 增到 399，其中“完全没有框”的翻了一倍。
3. **RegionDense 在结构上碰不到这些图。** 这些 aircraft 加上保护邻域后放不进 320 的格子，主图只能走原样的整图路径；它们也进不了补图库。训练集 94,493 张有目标的图里，31,573 张走整图回退。对这些图，E009 与 E008 的数据是一样的。
4. **RegionDense 的收益在它能处理的范围内是一致的。** 输入 64 像素以下的三档都高于 E008（+2.1、+1.0、+0.7），OGSOD tank 的 AP75 +2.2，MSAR 与 224 来源的 aircraft 继续上升。
5. **所以按“大目标”去调 RegionDense 的裁图或采样规则，方向不对。** 要补的是 Dense O2O 里对这一格 aircraft 起作用、而 E008 和 E009 都拿掉了的那一部分。Dense O2O 只有两个部件：Mosaic 和 MixUp。Mosaic 的缩小对其他 64 像素以上的类别是有害或无益的（去掉后 ship、bridge 都涨），所以更可能起作用的是混合带来的正则化；其中 MixUp 不缩小任何目标，与 RegionDense 的保尺度原则不冲突。这是推断，E011 用来检验它。

## E011：RegionDense + MixUp

- **类别：** 组件消融，同时是 Dense O2O 两个部件的拆分。
- **攻击的问题：** other_512 aircraft 在去掉 Dense O2O 后 AP 62.6 → 59.8、漏检 261 → 399；RegionDense 只恢复到 60.9。
- **假设：** 如果恢复 MixUp（概率 0.5，第 4–20 轮，与 E000 完全相同），other_512 aircraft 的检出会回升，因为它需要的是混合带来的正则化而不是缩小；同时 64 像素以下目标的收益保持，因为 MixUp 不改变目标尺度。
- **改动：** 只有配置。`configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_mixup_36e_b32.yml` 在 E009 上把 `mixup_prob` 从 0 改回 0.5，其余全部相同（`tests/test_region_dense_mixup_config.py` 检查只有这一项不同，且与 E000 的 MixUp 设置一致）。不新增代码、不新增参数，推理模型相同。
- **主指标：** 36 轮 val AP。对照是 E009（66.12）和 E008（65.86）。
- **机制指标（val，用本文的拆解工具）：**
  - other_512 aircraft 的 AP 与漏检数回到 E000 附近（62.6、261）；
  - 输入长边 < 64 的三档 AP 不低于 E009（39.4、54.7、69.9）；
  - OGSOD tank、OGSOD bridge 的 AP 不低于 E009（32.2、47.8）。
- **读法：**

| 结果 | 结论 |
|---|---|
| aircraft 回升，小目标保持，val AP 高于 E009 | 保留。Dense O2O 的改造定为“Mosaic 换成保尺度区域拼图，MixUp 保留” |
| aircraft 回升，但小目标下降 | MixUp 对 aircraft 有用、对低对比度小目标有害。下一步让 MixUp 只作用于对比度足够的图，而不是全局开关 |
| aircraft 不回升 | 起作用的是 Mosaic 的缩小或换背景。下一步对走整图回退、且全部目标 ≥ 64 像素的图恢复缩小拼图（训练集里这样的图有 24,592 张） |

- **风险：** MixUp 以约 0.5 的比例混合整幅图，会把低对比度小目标的对比度再减半。OGSOD tank 的对比度本来只有 0.2–0.3 dB，可能受损；这正是第二行要检查的。
- **代价：** 一次 36 轮训练，约 18 小时，GPU 4、5。

## 服务器命令（8 卡服务器，与对照 E008、E009 同一台）

E008、E009 都在这台服务器的 GPU 4、5 上运行，E011 要与它们直接比较，所以放在同一台。区域库沿用 E009 的 `sardet100k_train_region_dense_v1.npz`，不需要重新生成。

更新代码（工作区若有未提交改动或分叉，先查看，不要强制覆盖）：

```bash
cd /root/zxy/deim
conda activate deim_zxy
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
```

启动（只在 GPU 4、5 同时空闲时才会启动；内部用 `nohup ... > console.log 2>&1 < /dev/null &`，保存 `$!` 后 `disown`）：

```bash
cd /root/zxy/deim
conda activate deim_zxy
GPU_IDS=4,5 \
EXPERIMENT_ROOT=/root/zxy/experiments \
INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth \
RUN_PREFIX=deim_m_sardet100k_region_dense_mixup \
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_mixup_36e_b32.yml
```

重连后（把启动时打印的目录填进去）：

```bash
conda activate deim_zxy
export OUT=/root/zxy/experiments/替换为启动时打印的完整目录
tail -f "$OUT/console.log"
```

GPU 与进程：

```bash
nvidia-smi -i 4,5
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
```

指标（`log.txt` 每轮一行，数值乘 100；按实测 val AP 选权重，不只看阶段文件名）：

```bash
python - <<'PY'
import json, os
from pathlib import Path
rows = [json.loads(line) for line in (Path(os.environ['OUT'])/'log.txt').read_text().splitlines() if line.strip()]
for row in rows[-5:]:
    a = row['test_coco_eval_bbox']
    print(row['epoch'], dict(AP=round(100*a[0], 2), AP50=round(100*a[1], 2), AP75=round(100*a[2], 2), APs=round(100*a[3], 2), APl=round(100*a[5], 2)))
best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
print('best epoch', best['epoch'], round(100*best['test_coco_eval_bbox'][0], 2), 'best_stg2.pth' if best['epoch'] >= 32 else 'best_stg1.pth')
PY
```

TensorBoard：

```bash
tensorboard --logdir "$OUT/summary" --host 127.0.0.1 --port 6006
```

恢复（确认原进程已停止；不要同时传 `-t` 和 `-r`）。`last.pth` 只更新到无增强阶段之前，最后 4 轮中断会从阶段边界重跑：

```bash
cd /root/zxy/deim
conda activate deim_zxy
GPU_IDS=4,5 bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

训练结束后的 val 拆解（与本文同一口径，`$E` 为 E000 运行目录下的 `independent_verify_20261106`）：

```bash
E=/root/zxy/experiments/deim_m_sardet100k_deim_hgnetv2_m_sardet100k_36e_b32_nozoom_nocrop_seed0_20261105_125147_N2zAJh/independent_verify_20261106
D=$E/d11_region_dense_val
cd /root/zxy/deim && export PYTHONPATH=/root/zxy/deim
CUDA_VISIBLE_DEVICES=4 python $D/code/e1_report.py collect "$OUT" $E/report_E011 --device cuda:0 --batch 16 --checkpoint "$OUT/best_stg2.pth"
mkdir -p $D/E011 && cp $D/E009/val_source_groups.json $D/E011/
CUDA_VISIBLE_DEVICES= python $D/code/source_class_eval.py $E/report_E011 $D/E011
```

## 局限

- 单种子；E008 同时去掉了 Mosaic 和 MixUp，所以“aircraft 需要的是 MixUp”目前只是推断。
- 没有检验 aircraft 的下降是否集中在训练集里没有相似样本的目标上；若是，正则化的解释会更有依据。
- 拆解按 score ≥ .3 的工作点统计漏检，AP 不受这个阈值影响。
