# E016 / RegionDenseSAR v3：保护区域驱动的非等分拼图

状态：实现与功能验证阶段，尚未记录训练结果。正式运行按用户授权使用新服务器 GPU 4、5；启动前检查这两张卡是否空闲。本实验独立评价 v3，未叠加 E013-v2 等模型改动。

## 相对 E010 改了什么

E010 先尝试四格、等宽两列、等高两行，均不满足时退回整图。v3 保留这三条路径及顺序；只有它们全部失败时，增加非等分两列/两行。因此占据超过半幅、但不到整幅的目标及其邻域，可以与所需区域较小的源图共同训练。

所有坐标都在普通 resize 到 640 的坐标系内。每个 GT 的保护范围仍为四周各一个自身长边，超出原图的部分裁到原图边界。用同一源图所有保护范围的外包矩形 R 表示必需区域，其最小整数宽高为 `ceil(R.right/bottom) - floor(R.left/top)`。

两列布局要求 `wA + wB <= 640`，切分位置取合法范围中点 `floor((wA + 640 - wB) / 2)`；两行同理。每个源图裁块的起点也取合法范围中点。槽位随机交换时，同时交换对应宽高与切分位置。所有源图的全部 GT 和保护邻域必须保留，不缩小目标，也不通过丢弃目标满足最多 300 GT 的限制。

同一对图两种方向均合法时，使用实际裁块的背景距离较小者；在距离最近的至多 4 个不同补图中随机选择。背景特征沿用 E010 的灰度 25/50/75 分位数、相邻灰度差 50/90 分位数，排除 GT 及外扩 2 px，至少 256 个有效背景像素和差分。归一化直接复用 E010 的 train IQR（下界 1 灰度级），不重新拟合尺度。缺少有效背景特征时，在几何合法的补图和方向中随机选择。不能拼图的源图仍按整图使用。

背景特征针对**实际非等分裁块**计算。CPU 预处理采用逐行/列累计直方图，精确重现整数灰度的线性插值分位数，排除跨裁块边界的差分。离线缓存约 4.51 GiB 浮点特征和 0.11 GiB 有效性数组；训练仅加载小型几何库及固定 epoch 计划，不读取这两个大表、不在线搜索布局。

容量 64 的 FIFO 队列、0.5 组图尝试概率、完整源图遍历和普通全图训练均保持。每次完整源图遍历仍使所有源图及每个 GT 恰好出现一次；固定优化器步数下，拼图率提高会增加完整遍历次数，这是本方法增加有效监督密度的途径。

## 依据与判断

train-only 几何诊断：94,493 张图中 31,573 张不能进入任何 E010 固定布局。在双方均随机选择补图的同一次完整遍历中，非等分布局使其中 12,841 张能够参与组图，GT/画布从 3.1156 到 3.4336；aircraft 组图曝光比例从 21.09% 到 46.39%，tank 从 20.18% 到 45.43%。该诊断没有背景择图，不等同于正式训练计划，也不代表 AP 增益。

直接对照为 E010（val 66.45）和 E008（val 65.86）。用户希望最终相对 E008 达到约 +0.8–1.0 AP，对应 val 66.66–66.86；这是目标而非保证。按 val 评估总体 AP，并观察 aircraft、大目标是否得到改善及小目标收益是否保留；不根据 test 选择配置或 checkpoint。

## 固定训练协议

| 项目 | 设置 |
| --- | --- |
| 数据 | 官方 train 94,493 / val 10,492 / test 11,613，六类，原始标注 |
| 模型 | DEIM-D-FINE-M / HGNetv2-B2；原匹配、MAL、框回归与推理 |
| 初始化 | `/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth`，`-t`，新优化器 |
| 训练 | seed 0，36 轮，全局 batch 32，GPU 4、5 各 16 |
| 输入与步数 | 基准 640，batch 多尺度 480–800 到 epoch index 32；每轮 2,952 步 |
| v3 阶段 | epoch index 4–19，共 16 轮；每轮 94,464 张输出画布 |
| 优化器 | AdamW，主 LR 4e-4、backbone 4e-5、WD 1e-4 |
| 调度 | flatcosine，warmup 2000、flat 20、no-aug 4 |
| 其他增强 | 与 E010 一致；原 Mosaic/MixUp 均关闭 |
| EMA / AMP | 原 EMA 0.9999；FP16、FP32 loss、非有限批次一次 FP32 重算 |
| 评估 | 每轮完整 val，640；按 val AP 选择 checkpoint，test 后续独立评估 |

新配置只覆盖数据集类型、计划目录、`adaptive_partition` 与运行记录字段。既有 E010 配置及其计划不改动。

## 文件和验证

- `engine/data/region_dense_adaptive.py`：保护区域、精确背景分位数、非等分后备路径。
- `engine/data/dataset/region_dense_adaptive_dataset.py`：6 列计划渲染，沿用 E010 双卡采样与按 epoch 恢复。
- `tools/build_region_dense_adaptive.py`：只读 train、复用 E010 几何和归一化、生成 16 轮固定计划。
- `tools/audit_region_dense_adaptive.py`：全部计划的双卡覆盖与几何约束、实际像素与目标保留、缓存特征与实际裁块一致性。
- `tests/test_region_dense_adaptive.py`：关闭新路径时逐项复现 E010、完整源图遍历、背景择图、目标数限制、两方向及交换槽位、跨 worker/恢复一致、配置差异。
- `configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_adaptive_36e_b32.yml`：E016 独立配置。
- 数据集注册与 launcher 元数据增加 v3 标识；原启动、恢复及 GPU 空闲检查流程不变。

功能核对不代表模型收益。正式训练前在真实 train 上检查 200 张组图；单种子实验完成后才能判断这一版是否有效。

## 新服务器更新与准备

先查看 worktree，若已有改动阻碍更新则保留并解决，不强制覆盖。

```bash
cd /root/zxy/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
git status --short --branch
git fetch origin
git switch main
git -c pull.rebase=false pull --ff-only origin main
export CFG=configs/deim_dfine/deim_hgnetv2_m_sardet100k_region_dense_adaptive_36e_b32.yml
export BASE_PLAN=/root/zxy/experiments/region_banks/sardet100k_train_region_dense_sar_v2_seed0_b32
```

只准备一次，正式计划写入配置中的新目录，已完成时直接复用。准备过程可离线运行：

```bash
export PREP=$(mktemp -d /root/zxy/experiments/diagnostics/region_dense_sar_v3_prepare_XXXXXX)
nohup env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
python tools/build_region_dense_adaptive.py --config "$CFG" --base-plan-dir "$BASE_PLAN" \
  --workers 8 --world-size 2 > "$PREP/build.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PREP/build.pid"
disown "$PID"
tail -f "$PREP/build.log"
```

准备完成后检查真实训练数据构造：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest discover -s tests -p 'test_region_dense_*.py' -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/audit_region_dense_adaptive.py --config "$CFG" --output "$PREP/data_check" --pixel-checks 200
```

## 启动、日志与恢复

launcher 在启动前检查 GPU 4、5；二者都空闲才创建训练进程。每次正式新运行生成独立目录、保存解析配置、Git revision、协议和计划报告，使用 nohup 并保存 PID。

```bash
GPU_IDS=4,5 EXPERIMENT_ROOT=/root/zxy/experiments \
INITIAL_WEIGHTS=/root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth \
RUN_PREFIX=regiondense_sar_v3 \
bash tools/run_sar_module_experiment.sh "$CFG"
```

重连后激活环境，将 OUT 设置为启动输出中的完整目录：

```bash
conda activate deim_zxy
export OUT=/root/zxy/experiments/REPLACE_WITH_ACTUAL_V3_RUN
tail -f "$OUT/console.log"
nvidia-smi -i 4,5
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
tail -n 5 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 127.0.0.1 --port 6006
```

中断后恢复（原训练已停止时使用）：

```bash
cd /root/zxy/deim
conda activate deim_zxy
GPU_IDS=4,5 bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复使用保存的配置、`-r last.pth` 与固定 epoch 计划，不能同时传 `-t`。当前 solver 的 `last.pth` 只在 no-augmentation 阶段前更新，不代表最终轮；计划按 checkpoint 保存的 epoch 边界恢复。

## 指标与最终 test

训练 JSONL 指标在 `log.txt`，TensorBoard 在 `summary/`，COCO 对象在 `eval/latest.pth`。bbox 数组前六项依次为 AP、AP50、AP75、APs、APm、APl，乘 100 报告。

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['OUT'])
rows = [json.loads(x) for x in (root/'log.txt').read_text().splitlines() if x.strip()]
for row in rows[-5:]:
    print(row['epoch']+1, dict(zip(['AP','AP50','AP75','APs','APm','APl'],
                                  [100*v for v in row['test_coco_eval_bbox'][:6]])))
best = max(rows, key=lambda r:r['test_coco_eval_bbox'][0])
stage = 'best_stg2' if best['epoch'] >= 32 else 'best_stg1'
(root/'selected_val_checkpoint.txt').write_text(str(root/(stage+'.pth'))+'\n')
print('val-selected:', best['epoch']+1, 100*best['test_coco_eval_bbox'][0], stage)
PY
```

完成 36 轮、上述命令按 val 选好权重后，独立评估 test（先确认 GPU 4、5 空闲）：

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
