# SAR-AIRcraft：协议 v2（训练 trick 调整与 A/B 训练方案）

日期：2026-11-03。依据：[协议与训练设置调研](sar_aircraft_protocol_survey.md)、[型号混淆线索诊断](sar_confusion_cues_diagnosis.md)。
旧的冻结 102 轮协议（640 输入，全局 batch 28）及其全部结果保持不变；v2 用新配置，Δ 只在同一协议内计算。

## 1. 训练集合：推荐 A→B

所有方案都在官方 test（879 张，2950 个目标）上报告，与文献一致。

- **A（开发与消融）**：train 3047 张训练，官方 val 选权重，与 PGD 相同。v2 的各阶段筛选、模块消融都在 A 下做，
  已有 640 结果可直接作对照。
- **B（论文最终表）**：train + val（3489 张，13513 个目标）训练，与官方识别实验、SAR-Net 的"默认划分"相同，
  轮数用 A 下定好的值，取最后一轮权重，不看 test 选权重。论文主表同时给出 A 与 B。

**B 不能照搬 A 的"最佳轮次"。** DEIM solver 的训练轨迹本身依赖 val：进入无增强阶段（epoch 90）时载入 val 最优的
`best_stg1.pth`，之后每当 val 没有提升就回退到 `best_stg1.pth` 并降低 EMA decay。所以 A 中第 N 轮的权重不是
"训练 N 轮"的结果，而是 val 选择过的轨迹；B 里 val 已并入训练，这套机制失去意义，按轮次对齐也对不上。

B 的做法：

1. 2026-11-03 起 solver 在最后一轮训练结束、val 回退之前保存 `final.pth`（不改变训练）。阶段 1 起的 A 运行都会有它。
2. v2 定型后，在 A 下用"固定日程"（关闭 epoch 90 的载入与无提升回退，需新增开关，默认关闭以保持现有行为）跑一组，
   比较 test 上 `final.pth` 与 val 选出的权重。差距小（<0.3 AP）→ 固定日程取最后一轮是可靠的选择规则。
3. B 用同样的固定日程训练 train + val，取 `final.pth` 报告 test；另需合并 train/val 标注与 `stats.json`。

这样 B 的权重选择不依赖任何评估集合，规则在 A 中预先验证，可以在论文中直接说明。不保证取到 test 上最好的那一轮，
但任何不看 test 的规则都无法保证；A 中 val 选权重本身也只有 Spearman 0.63 的排序一致性。

## 2. 分阶段调整（均为方案 A，单因素）

| 阶段 | 变量 | 配置 | 对照 | 采用条件 |
|---|---|---|---|---|
| 1（本次） | 输入 640 → 1024 | `deim_hgnetv2_m_sar_aircraft_1024_102e_b12_old_server.yml`；随后 `..._rem_1024_...` | 640 baseline：val 64.51 / test 45.74（AP50 65.37）；640 REM：test 47.31 | test AP 比 640 baseline 高 ≥1.5 → 采用 1024；0–1.5 → 再试 800 折中；<0 → 保留 640 |
| 2 | 增强：去掉 `RandomPhotometricDistort`；另一组去掉 `RandomHorizontalFlip` | 阶段 1 结果出来后再写 | 阶段 1 选定的 baseline | val 与 test 同向变好才采用（单种子，差异 <1 AP 视为不确定） |
| 3 | 轮数 102 → 约 150 | 同上 | 阶段 2 选定配置 | 同上 |
| 4 | 冻结 v2；重跑 REM 等候选模块；关键结论补 2–3 个种子 | — | v2 baseline | — |
| 5 | 方案 B 最终训练 | — | — | — |

阶段 1 只改分辨率以及它直接带来的变化，其余全部继承冻结 102 轮协议（`tests/test_sar_protocol_v2.py` 检查）：

| 项目 | 640（冻结） | 1024（v2 阶段 1） | 说明 |
|---|---|---|---|
| 训练 / 验证尺寸 | 640；多尺度 480–800 | 1024；多尺度 768–1280 | `generate_scales` 同一规则 |
| Mosaic 输出 | 320 | 512 | 保持为输入的一半 |
| 全局 batch | 28（每卡 14） | 12（每卡 6） | 显存约随像素数增长 2.6 倍 |
| 主 LR / 骨干 LR | 3.5e-4 / 3.5e-5 | 2.3e-4 / 2.3e-5 | AdamW 按 sqrt(12/28) 缩放 |
| warmup | 2000 iter（约 18.3 轮） | 4660 iter（同样约 18.3 轮） | 每轮迭代数 109 → 254 |
| 验证 batch | 64 | 32 | 只影响显存 |

不变：DEIM-D-FINE-M，COCO M 权重 `-t`，seed 0，102 轮，flat 49、no_aug 12，policy [4,49,90]，mixup [4,49]，stop 90，
WD 1e-4，EMA 0.9999，AMP FP16 + FP32 loss/retry，关闭 ZoomOut/IoUCrop，按 val AP 选权重并报告 test。

batch 与 LR 随分辨率一起变，阶段 1 的差异不完全等于分辨率本身的作用；这是显存约束下的必要代价，在论文中如实说明。
每卡 6 张在 1280 尺度下的显存没有实测；若首个多尺度 epoch 出现 OOM，告知后改为全局 batch 8 的配置。

## 2.1 阶段 1 结果：1024 baseline（2026-11-03）

运行：`/home/zxy/sar/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_1024_102e_b12_old_server_seed0_20261103_135917_0hVcJe`。
训练 102 轮正常结束，无 OOM，PyTorch 峰值显存 13.4 GB/卡，约 2.5 分钟/轮。

| | val AP | test AP | AP50 | AP75 | APm | APl | AR100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 640 baseline（v1） | 64.51 | 45.74 | 65.37 | 51.42 | 46.82 | 43.11 | 68.92 |
| 1024 baseline，best_stg2 / epoch 100 | 62.38 | 45.00 | 63.90 | 50.62 | 44.77 | 48.70 | 70.60 |
| 1024 baseline，final.pth / epoch 101 | 62.26 | 45.0 | 63.8 | 50.6 | 44.8 | 48.7 | 70.6 |

test 各类 AP（1024 / 640 配对诊断导出口径，后者约高 0.4）：A220 39.21 / 37.30，A320/321 61.66 / 62.08，A330 79.04 / 77.54，
ARJ21 35.03 / 37.11，Boeing737 22.07 / 23.29，Boeing787 34.56 / 42.43，other 43.45 / 42.96。

**判定：按规则（test < 640 baseline）不采用 1024，但该运行没有收敛，比较不公平。**

- val AP 到最后仍在上升：80–89 轮均值 60.71，90–101 轮 62.05，最后几轮每轮仍涨约 0.05–0.1；最佳在第 100 轮。
- 起步很慢：第 15 轮 val AP 5.58，第 25 轮 39.13。warmup 4660 iter 约到第 18 轮才结束，主 LR 又按 sqrt 降到 2.3e-4。
- AR100 更高（70.60 对 68.92）、APl 更高，AP50 更低，787 降 7.9：框找到得更多，型号判别没有变好。
- `final.pth` 与 val 选出的 best_stg2 在 test 上几乎相同（45.0 对 45.00），是"固定日程取最后一轮"规则的第一个支持点。

**与 640 同轮次 val 曲线对比：1024 是训练不足，不是分辨率无益的定论。**

| 轮次段 | 640 val AP 均值 | 1024 val AP 均值 |
|---|---:|---:|
| 40–49 | 57.15 | 50.50 |
| 60–69 | 64.03 | 56.90 |
| 80–89 | 64.44 | 60.71 |
| 90–101 | 64.47（第 60 轮后基本走平） | 62.05（仍在上升） |

两者起步同样慢（warmup 都约 18 轮，第 15 轮 val 分别为 8.55 / 5.58），差别出现在第 25 轮之后：640 在约第 65 轮走平，
1024 每段仍涨 1–3 AP。原因最可能是 batch 12、主 LR 2.3e-4 下收敛更慢，而不是 1024 本身。

**阶段 1b：800 输入，全局 batch 20**（`deim_hgnetv2_m_sar_aircraft_800_102e_b20_old_server.yml`）。按 1024 运行的峰值显存
（13.4 GB/卡，每卡 6 张，最大 1280）按像素换算，800 的最大尺度 992 下每卡 10 张约同量级。batch 与 LR 更接近冻结协议
（LR 3.0e-4，sqrt(20/28) 缩放；warmup 2810 iter 仍约 18.3 轮；Mosaic 400；多尺度 576–992），干扰更少，速度约为 1024 的一半多。

判定（对照 640 baseline test 45.74）：test ≥ 47.24 → 采用 800；45.74–47.24 → 看 val 曲线是否走平，未走平则阶段 3 延长轮数时
同时比较 640 与 800；< 45.74 → 保留 640，进入阶段 2。1024 暂不再跑，若 800 有收益且未收敛，再考虑 1024 + 更长日程。

另一个共性问题：两种分辨率前 15 轮 val 都很低，18 轮 warmup 占了训练的 1/6。缩短 warmup 是潜在 trick，放到阶段 2/3 单独检验。

## 2.2 策略调整（2026-11-03）：停止逐项调分辨率，主协议改为方案 B

用户判断：640/1024 下 test AP50 都只有约 64–67，比文献的 85–90 低一档，逐项试下去效率太低。已有证据的结论：

1. **差距不在检测能力本身。** 我们的 val AP50 约 88（640），与文献 RT-DETR 87.6、YOLOv8 88.7、DINO 85.2 同档；
   test 上类别无关 AP50 81.9，高于官方单类结果 71.6；train 中见过的 test 目标型号正确率 0.94。DEIM-D-FINE 不是瓶颈。
2. **差距来自训练集合与评估口径。** test 中没在 train 出现过的目标只有约 0.41 的型号正确率；官方识别实验与 SAR-Net 用
   官方 trainval（13513 个目标）训练，若 val 含有 test 飞机的重叠切片，用 trainval 训练会让 test 中"见过"的比例上升。
   部分文献只写"mAP"，评估集合不明，也可能是在 val 上。我们一直在最难的口径（只用 train，测 test）下比较。
3. **训练配方已收敛。** 640 冻结配方在约第 65 轮走平，`final.pth` 与 val 选权重在 test 上一致；延长轮数或提高分辨率都没有改善 test。

**决定**

- **主协议：方案 B**——官方 trainval 训练（数据集自带 `COCO/trainval.json`，即 ImageSets 的 trainval），官方 test 报告，
  640 冻结配方，102 轮，`fixed_schedule: True`（关闭 val 驱动的 epoch 90 载入和 stage-2 回退），取 `final.pth`。
  这与官方数据集论文的"训练集 13513 / 测试集 2950"和 SAR-Net 的"默认划分"一致，不需要选权重集合。
- 方案 A 保留为诊断口径（已有结果不作废）；分辨率/增强/轮数的逐项搜索停止。800 运行若已开始可跑完作参考，未开始则不跑。
- 先用 CPU 跑 `tools/check_split_overlap.py`，量化 test 中"train 没有、val 有"的目标数，预估方案 B 的提升并解释文献差距。
- 运行顺序：B baseline → B REM → 在 B 上做方法（斑点重采样）。若 B baseline 的 test AP50 仍明显低于 80，
  再在同一协议下复现一个外部基线（如 Ultralytics YOLOv8）判断是否是 DEIM 配方问题。

**划分重叠检查结果（2026-11-03，`/data2/zxy/sar/experiments/diagnostics/split_overlap_*`，阈值 0.9）：
val 与 test 重叠 73.3%，这是文献分数高的主要来源。**

| 查询 | 参照 | 有匹配比例 |
|---|---|---:|
| val | train | 81.8% |
| test | train | 50.3% |
| test | val | 73.3% |
| test | train + val | 77.4%（其中 797 个只在 val 中出现） |

| test 类别 | GT | train 中见过 | train + val 中见过 |
|---|---:|---:|---:|
| Boeing737 | 550 | 128（23%） | 398（72%） |
| A220 | 460 | 252（55%） | 357（78%） |
| ARJ21 | 362 | 179（49%） | 270（75%） |
| Boeing787 | 454 | 275（61%） | 341（75%） |
| other | 1041 | 568（55%） | 833（80%） |
| A320/321、A330 | 52、31 | 全部 | 全部 |

val 与 test 来自相同场景的重叠切片：用 trainval 训练后，test 中"见过"的目标从 50% 升到 77%，最难的 737 从 23% 升到 72%。
按 A 口径下的点统计（见过的目标定位且判对约 0.89，未见过约 0.22）粗算，加权比例从约 0.56 升到约 0.74；
这只是方向性估计，不等于 AP，实际以 B baseline 的 test 结果为准。它同时解释了：test 上 737 最差、val 与 test 的大差距、
以及按官方 trainval 训练的文献（官方识别实验、SAR-Net）test 分数远高于我们 A 口径的原因。方案 B 是与文献可比的口径。

配置：`deim_hgnetv2_m_sar_aircraft_trainval_102e_b28_old_server.yml`（只改训练标注与 `fixed_schedule`），
`deim_hgnetv2_m_sar_aircraft_rem_trainval_102e_b28_old_server.yml`（只多 REM）。训练中每轮仍在 val 上评估，
但 val 已在训练集内，这些数字只用于观察曲线，不用于选权重；`best_stg*.pth` 不使用。

## 3. 服务器命令（旧服务器）

更新代码与环境：

```bash
cd /home/zxy/sar/repos/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
```

工作区有改动或分叉时先查看，不要 reset。数据已就绪，不需要准备。

启动阶段 1 baseline（两卡空闲时；启动器检查 GPU、新建目录、保存解析配置 / Git revision / 协议，
以 `nohup ... > console.log 2>&1 < /dev/null &` 启动并保存 PID 后 `disown`，打印 `STARTED OUT=...`）：

```bash
nvidia-smi -i 0,1
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_1024_102e_b12_old_server.yml
```

阶段 1b（800）启动：

```bash
bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_800_102e_b20_old_server.yml
```

（1024 已完成；下面的 1024 REM 暂缓，等分辨率定下来再跑对应分辨率的 REM。）

baseline 结束后启动 REM（同样的环境变量）：

```bash
bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_1024_102e_b12_old_server.yml
```

监控（重连后先设置 `OUT` 并激活环境）：

```bash
export OUT=/home/zxy/sar/experiments/替换为启动器打印的目录
tail -f "$OUT/console.log"
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
nvidia-smi -i 0,1
tail -n 3 "$OUT/log.txt"
python - "$OUT/log.txt" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
for r in rows[-3:] + [best]:
    v = r['test_coco_eval_bbox']
    print(r['epoch'], 'val AP', round(v[0] * 100, 2), 'AP50', round(v[1] * 100, 2), 'AP75', round(v[2] * 100, 2))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

只在进程已停止时恢复：

```bash
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

`last.pth` 只在 epoch < 90 时更新，不是最终权重；`best_stg1.pth` / `best_stg2.pth` 按 val AP 保存，选择时以日志中最佳 val AP 的轮次为准。

训练完成后按 val 选权重评估 test（评估沿用该运行自己的 1024 配置）：

```bash
export TESTBATCH=/home/zxy/sar/experiments/test_evals_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
ulimit -Sn 65535
nohup python tools/run_sar_test_evals.py --run "$OUT" \
  --data-root /home/zxy/sar/datasets/SAR-AIRcraft-1.0 --output "$TESTBATCH" \
  > "$TESTBATCH.console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$TESTBATCH.pid"
disown "$PID"
tail -f "$TESTBATCH.console.log"
cat "$TESTBATCH/summary.md"
```

把 `summary.md` 和最佳 val 轮次贴回即可。

## 4. 本地验证

Windows Conda `pytorch`（CPU）：`tests/test_sar_protocol_v2.py` 解析两个新配置，检查 1024 baseline 相对冻结 102 轮协议
只改了上表中的项目、多尺度范围为 768–1280、LR 与 warmup 的换算，以及 REM 版本只多 `use_rem`。未在服务器运行，没有训练结果，
显存占用未知。
