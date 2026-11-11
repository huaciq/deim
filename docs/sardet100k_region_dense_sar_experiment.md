# RegionDenseSAR v2：正式实现与训练协议

本实验在用户授权的新服务器 GPU 4、5 上运行。直接对照 E008（关闭原 Dense O2O）和 E009（RegionDense v1）；原始 DEIM E000 保留为参考。诊断动机见 [后续设计](sardet100k_region_dense_sar_followup.md)。

## 当前运行 E010

2026-11-11 21:49:50（北京时间）已启动，训练代码 `33172bb`，launcher PID `1811655`。已观察到 epoch index 0 的第 100/2952 步，双卡均参与计算，当前损失有限、FP32 retry 为 0；尚无完整 epoch 的 val 指标。

运行目录：`/root/zxy/experiments/regiondense_sar_v2_deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32_seed0_20261111_214950_thYezk`。

Windows pytorch 和 Linux deim_zxy 各通过 11 项功能检查。全训练集背景特征和 16 轮计划已生成，用时 292.5 秒；每轮 2,952 步，增强阶段平均 3.1126 GT/输入，23 次完整源图遍历，实例出现 23–24 次。200 张实际组图核验通过，全部源图 GT 保留、像素与普通缩放后裁块一致，框边长最大数值差 5.68e-14 px，后续 Sanitize 删除数为 0。与 E000/E008/E009 实际保存的配置比较，只有本方案的数据构造及运行记录字段变化。

CPU 报告、协议对比、功能检查日志和启动记录：`/root/zxy/experiments/diagnostics/region_dense_sar_prepare_20261111_NicyRp/`。以上为启动时状态，后续指标以运行目录 log.txt 为准。

## 方法与固定设置

- 保持普通缩放至 640 后的目标尺度；每个目标周围保留一个自身长边的邻域，边界裁到原图范围。
- 主图和补图都保留全部 GT 与受保护邻域；合法裁块取原点区间中点。布局依次尝试四格、两列、两行，不能装入则用整图。每个拼图最多 300 GT，不通过删标注适配 query 数。
- 训练原图随机排列，使用容量 64 的 FIFO 等待队列。先取最早进入的源图，以 0.5 概率尝试组图；已使用的原图移出队列。每完整遍历一次源图，每张图及其中每个 GT 恰好出现一次。空标注图及不能裁切的图也正常使用。
- 裁块背景去除 GT 框及框外 2 px，计算灰度 25/50/75 分位数和相邻像素灰度差 50/90 分位数；至少 256 个有效背景像素及差分。使用整个 train 的特征 IQR 归一化，下界为 1 灰度级。
- 从几何合格且背景距离最近的至多 4 张图中随机选补图；特征不可用时随机选。各源图最终都会被使用，因此背景选择不筛除数据来源。槽位随机打乱，像素不做亮度匹配或去斑处理。
- 这是显示图像统计相容性，不声称恢复定标散射、斑点参数或完整阴影。

离线只读取 train，生成背景特征库和 epoch index 4–19 的组图计划。源图流跨增强 epoch 连续推进；每轮保持 baseline 输出画布和优化器步数。对 94,493 张图、双卡 global batch 32，仍为每轮 94,464 张输出画布、2,952 步。完成的源图遍历严格保留实例分布，最后一遍未完成的部分有有限抽样波动。

训练按固定计划分配给 DDP ranks，worker 只读当前 epoch 的计划；普通阶段沿用 baseline DistributedSampler。恢复到下一个 epoch 时直接读取对应计划，无需在各 worker 中恢复随机队列。当前 solver 的 checkpoint 是 epoch 边界，未新增 batch 中途恢复功能。

其余协议与 E008 一致：seed 0；36 轮；global batch 32，GPU 4、5 各 16；基准输入 640，batch 多尺度 480–800 到 index 32；原 Hungarian 匹配、MAL 和框回归；COCO DEIM-D-FINE-M 90e 权重初始化；LR 4e-4、backbone 4e-5；warmup 2000、flat 20、no-aug 4；原 EMA 和 FP16 AMP/FP32 loss/retry。原 Mosaic 与 MixUp 均关闭。val/test 数据和推理模型不变。

## 实现与检查

- `engine/data/region_dense_sar.py`：背景描述及保持源图遍历的组图器。
- `engine/data/dataset/region_dense_sar_dataset.py`：固定计划数据集、双卡采样、按 epoch 恢复。
- `engine/data/region_composition.py`：共享的实际像素/框组合；v1 改为调用相同实现，行为保持。
- `tools/build_region_dense_sar.py`：train 特征、16 轮计划、统计报告。
- `tools/audit_region_dense_sar.py`：全部增强 epoch 的 DDP 覆盖与真实像素/目标检查。
- `tests/test_region_dense_sar.py`：源图完整遍历、不可裁图/空标注图、背景选择、双卡分片、跨 worker 与恢复一致性、其余配置与 E008 一致。

功能检查及已观察到的训练步骤不代表 36 轮训练完成。模型收益以最终 val/test 为准。先用 val 选择 checkpoint，test 不参与选参数。

## 新服务器命令

先检查已有改动；如阻碍同步，保留并解决，不强制覆盖。当前工作分支为 main。

```bash
cd /root/zxy/deim
conda activate deim_zxy
git status --short --branch
git fetch origin
git switch main
git -c pull.rebase=false pull --ff-only origin main
export CFG=configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32.yml
```

只准备一次；产物目录必须是新的。既有正式计划不覆盖，已完成时直接复用。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/build_region_dense_sar.py --config "$CFG" --workers 8 --world-size 2
export AUDIT=/root/zxy/experiments/diagnostics/region_dense_sar_impl_$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/audit_region_dense_sar.py --config "$CFG" --output "$AUDIT" --pixel-checks 200
```

正式启动（launcher 新建目录，nohup 后保存 PID 并 disown；不停止其他任务）：

```bash
GPU_IDS=4,5 EXPERIMENT_ROOT=/root/zxy/experiments \
INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth \
RUN_PREFIX=regiondense_sar_v2 \
bash tools/run_sar_module_experiment.sh "$CFG"
```

把实际启动输出的目录设为 OUT。重连需重新激活环境并设置 OUT。

```bash
conda activate deim_zxy
export OUT=/root/zxy/experiments/regiondense_sar_v2_deim_hgnetv2_m_sardet100k_region_dense_sar_36e_b32_seed0_20261111_214950_thYezk
tail -f "$OUT/console.log"
nvidia-smi -i 4,5
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
tail -n 5 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 127.0.0.1 --port 6006
```

停止后恢复（先确认原运行已停止；读取该实验的 resolved_config，使用 -r）：

```bash
GPU_IDS=4,5 bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

`last.pth` 只在无增强阶段之前更新，不能当作最后一轮。最终权重另存 `final.pth`；评估仍按 val AP 选择 best_stg1/best_stg2，不按文件名猜测。

查看最近指标并选出 val 最佳 checkpoint：

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['OUT'])
rows = [json.loads(x) for x in (root/'log.txt').read_text().splitlines() if x.strip()]
for row in rows[-5:]:
    a = row['test_coco_eval_bbox']
    print(row['epoch']+1, dict(zip(['AP','AP50','AP75','APs','APm','APl'], [100*v for v in a[:6]])))
best = max(rows, key=lambda r:r['test_coco_eval_bbox'][0])
stage = 'best_stg2' if best['epoch'] >= 32 else 'best_stg1'
(root/'selected_val_checkpoint.txt').write_text(str(root/(stage+'.pth'))+'\n')
print('val-selected:', best['epoch']+1, 100*best['test_coco_eval_bbox'][0], stage)
PY
```

训练结束后的 test 命令（上述命令先生成 val 选择结果；新建评估目录）：

```bash
export CKPT=$(cat "$OUT/selected_val_checkpoint.txt")
export EVAL=$(mktemp -d "$OUT/test_eval_selected_XXXXXX")
nohup env CUDA_VISIBLE_DEVICES=4,5 PYTHONUNBUFFERED=1 \
torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$CKPT" --test-only --seed 0 \
  --output-dir "$EVAL" --summary-dir "$EVAL/summary" \
  -u val_dataloader.dataset.img_folder=/root/zxy/TCRA-SAR/datasets/SARDet_100K/JPEGImages/test \
     val_dataloader.dataset.ann_file=/root/zxy/TCRA-SAR/datasets/SARDet_100K/Annotations/test.json \
  > "$EVAL/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$EVAL/launcher.pid"
disown "$PID"
tail -f "$EVAL/console.log"
```

COCO bbox 数组 0、1、2 是 AP、AP50、AP75，乘 100 报百分数。训练逐轮指标在 log.txt、TensorBoard 在 summary/、训练评估对象在 eval/latest.pth；单独 test 的控制台输出包含总体和分类指标。
