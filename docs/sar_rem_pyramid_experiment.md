# SAR-AIRcraft：主模块路线研判、test 诊断与 REM-Pyramid 实验

日期：2026-11-02。依据：[结果总表](sar_results_ledger.md)中 26 组旧服务器运行的 val 与 test。
目标（用户设定）：主模块 test AP +1.5～2.0；其余模块各 +1.0 以上。单种子，不预先做多种子。

## 1. 研判

**主模块围绕 REM 做。** 它是唯一在 58/102 轮、val/test 四个口径上都为正的模块
（test +0.92 / +1.57）；三项结构消融（space-to-depth −1.93、方形卷积 −1.07、point-strip −0.34）
都低于原 REM，说明 Haar 子带和方向条带两部分都起作用。102 轮 test 已达 +1.57，
处在目标区间下沿；58 轮 +0.92 还差。改进方向要回答"REM 的增益还能从哪里来"。

REM 的 test 增益集中在 AP75（+1.63）和大目标（APl +5.92，102 轮），但 REM 只作用于 stride-8。
大目标主要由 stride-16/32 的输出负责，现在只能通过 PAN 自下而上路径间接得到 REM 的增强。
因此第一个改进是 **REM-Pyramid**：在 stride 8/16/32 三个投影层各接一个独立 REM。
这是结构上最直接、且能形成"多尺度方向子带增强"叙述的扩展；s8+s16 版本作为它的层级消融。

**其余模块按"叠加在 REM 上"筛选。** 单独有效、叠加后失效的模块对论文没有用：
REM+CLCR-Lite（−0.20）、REM+W-FDR（−0.47）、REM+PACFG（+0.12）都低于 REM 单模块。
以后第二模块的实验单位是 REM+X，对照是 REM，而不是 baseline。

| 模块 | 处理 | 依据 |
|---|---|---|
| CLCR-Lite | 保留为第二模块候选，等诊断结果决定改法 | test +1.24 / +0.69；但与 REM 叠加为 −0.20 |
| OST-DA v2 | 暂留 | 58 轮 test +0.64，作用在 decoder 采样，与 REM 不在同一位置 |
| W-FDR | 停止 | 58 轮 test +0.65，102 轮 −1.53，不稳定 |
| PGD-REM、point-strip、space-to-depth、square | 停止，作为 REM 消融保留 | 都低于原 REM |
| PACFG、SDCP、SPI、BCCQ、CrossGate、PAFP、OST-DA v1、CLCR | 停止 | test ≤ 0 或明显为负 |
| SCS、DSG、SLQC、MDCC | 停止 | 102 轮 val 全部低于 baseline |

**需要一次 test 诊断，而且只需推理。** val 与 test 的差距在所有运行中一致（约 17–19 AP，AP50 约 88→67），
说明 test 的难点是系统性的。这个缺口来自"框找不到"还是"框找到了但型号判错"，
直接决定第二模块该做定位还是做细粒度型号判别；CLCR-Lite 在 test 上的收益大于 val，
也提示型号分数可能是 test 的主要短板。诊断用已有 baseline 与 REM 的 102 轮权重，
两次 879 张推理，不训练。

## 2. test 诊断（推理与 CPU 分析）

工具：`tools/compare_rem_102e.py` 新增 `--split test`，原 val 行为与旧输出不变。
报告在原有配对分析之外新增：

- **类别无关 AP**：7 类合并为 1 类，同一预测框只保留最高类别分数，避免 query×class 展平的重复框；
  缺口 = 类别无关 AP − 原 AP，表示型号错误或型号分数排序造成的损失。
- **定位与型号**：每个 GT 取 IoU≥0.5、score≥0.3 的任意类别预测中分数最高者，统计定位率、
  定位后型号正确率和完整混淆计数（`confusion_baseline.csv` / `confusion_rem.csv`）。不是一对一匹配。
- **各类 AP 表**写进 report.md。

判读：

| 观测（test，baseline） | 结论 | 第二模块方向 |
|---|---|---|
| AP50 缺口 ≥ 10 且定位率高 | test 下降主要是型号判别 | 细粒度型号判别 / 分类校准（CLCR-Lite 的改法） |
| AP50 缺口小、定位率低 | 目标没被找到 | 召回与特征（继续在 REM 一侧） |
| AP50 缺口小、AP75 缺口大 | 型号对但框不准 | 定位精度（FDR 一侧） |

诊断同时对 val 再跑一次（新目录，不改动已有 val 诊断输出），用来比较两个划分的缺口。

### 2.1 诊断结果（2026-11-02，102 轮 baseline 与 REM）

来源：服务器 `rem102_comparison_test_*` 与 `rem102_comparison_val_*` 的 report.md
（用户下载到 `C:\project\实验记录\report\`）。导出 batch 为每卡 8，test AP 比 `--test-only` 高约 0.4
（baseline 46.10 对 45.74），只在同一导出口径内比较。

**val→test 的 AP50 下降一半来自找不到目标，一半来自型号判错。**

| baseline | val | test | 下降 |
|---|---:|---:|---:|
| AP50（原） | 88.06 | 65.71 | −22.35 |
| AP50（类别无关） | 91.74 | 81.85 | −9.89 |
| AP50 缺口（型号） | 3.68 | 16.14 | +12.46 |

类别无关 AP / AP75 的"缺口"在 val 上为负，因为合并成一类后按全部目标统计，而原 AP 是 7 类平均，
两者权重不同；只有 AP50 缺口可近似当作型号损失。

**型号损失集中在窄体机。** test 上定位后型号正确率（score≥0.3，IoU≥0.5）：

| 类别 | test GT | test AP | val AP | test 型号正确率 | val 型号正确率 | test 定位率 | val 定位率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Boeing737 | 550 | 23.29 | 59.76 | 0.473 | 0.895 | 0.573 | 0.794 |
| A220 | 460 | 37.30 | 57.41 | 0.687 | 0.925 | 0.798 | 0.915 |
| ARJ21 | 362 | 37.11 | 57.90 | 0.693 | 0.858 | 0.837 | 0.966 |
| Boeing787 | 454 | 42.43 | 72.43 | 0.800 | 0.941 | 0.826 | 0.973 |
| other | 1041 | 42.96 | 56.49 | 0.865 | 0.934 | 0.728 | 0.892 |
| A320/321 | 52 | 62.08 | 75.47 | 1.000 | 1.000 | 0.981 | 0.988 |
| A330 | 31 | 77.54 | 72.09 | 1.000 | 1.000 | 0.839 | 0.889 |

test 中 Boeing737（19%）和 A220（16%）占比都高于 val（13%、13%），而这两类正是型号最容易错的。

**REM 的 test 增益主要来自两个小类。** 7 类平均 +1.54 中，A320/321（+4.75，52 个 GT）与 A330（+5.33，31 个 GT）
贡献约 +1.44；A220 +3.94；ARJ21、Boeing737、Boeing787、other 合计 −3.2 个类别点。
REM 提高了 Boeing737（0.473→0.533）与 Boeing787（0.800→0.835）的型号正确率，但 ARJ21、Boeing787 的定位率下降。
宏平均 AP 中每类权重相同，小类波动会放大整体结果，评估 REM-Pyramid 时同时看各类 AP。

**结论：第二模块做窄体机型号判别。** 型号缺口 12.5 个 AP50 点，是最大的可回收空间。
一个可直接检验的原因：SAR-AIRcraft 源图为 800/1000/1200/1500 正方形，全部缩放到 640，
同一型号的输入尺寸相差最多 1.9 倍，飞机的物理尺寸线索被打乱；而 CLCR-Lite 的输入里正好有 640 空间的
`log_w / log_h`，它在 test 上有收益。`tools/analyze_type_confusion.py` 用已有诊断目录的预测（只用 CPU）检验：

- 按源图尺寸分组的 AP、类别无关 AP50、定位率、型号正确率；
- 各类框尺寸在源图像素与 640 输入像素下的分布（IQR/中位数越小越能区分型号）；
- 最大的型号混淆对。

判读：型号正确率随源图尺寸明显变化，且源图像素下各类尺寸更集中 → 第二模块做"源图尺度感知的型号判别"
（把预测框在源图坐标下的尺寸送入分类，作为 CLCR-Lite 的升级）；两者都不成立 → 下降来自外观/场景差异，
改做外观特征层面的型号判别。

```bash
cd /home/zxy/sar/repos/deim
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
python tools/analyze_type_confusion.py \
  /data2/zxy/sar/experiments/diagnostics/替换为rem102_comparison_test目录 \
  /data2/zxy/sar/experiments/diagnostics/替换为rem102_comparison_val目录
```

只用 CPU，不影响正在运行的 REM-Pyramid 训练。结果在每个诊断目录的 `type_confusion/report.md`。

## 3. REM-Pyramid 实现与配置

`HybridEncoder.rem_feat_indices`（默认 `[0]`）选择接 REM 的投影层；索引 0/1/2 对应 stride 8/16/32。

- 索引 0 仍是原 `encoder.rem`，参数名、初始化与已报告 REM 完全相同；其它层是 `encoder.rem_levels.{i}` 中的独立 REM，
  结构、variant、residual mode 与原 REM 相同，各自 alpha 零初始化（初始整体为恒等）。
- 位置与原 REM 一致：输入投影（及该层 AIFI）之后、FPN/PAN 融合之前。stride-32 层的 REM 作用在 AIFI 输出上。
- 新层在 `fork_rng` 内构建，检测器其余参数初始化与 REM 单模块运行相同。默认配置的 state_dict 键不变。
- 训练日志新增 `train_rem_p4_alpha`、`train_rem_p5_alpha`（原 `train_rem_alpha` 仍是 stride 8）。
- 开销：每个 REM 2,823,681 参数；640 输入下 stride 8/16/32 各约 4.52 / 1.13 / 0.28 GMACs。
  Pyramid 相对 REM 增加 5.65M 参数、约 1.41 GMACs。

| 配置 | `rem_feat_indices` | 用途 |
|---|---|---|
| `deim_hgnetv2_m_sar_aircraft_rem_pyramid_58e_b28_old_server.yml` | [0, 1, 2] | 主实验 |
| `deim_hgnetv2_m_sar_aircraft_rem_s8s16_58e_b28_old_server.yml` | [0, 1] | 层级消融，主实验有效后再跑 |

两者都只继承旧服务器 REM 58 轮配置并改 `rem_feat_indices`、输出目录与本文档路径，训练协议完全相同：
DEIM-D-FINE-M，COCO M 权重 `-t`，seed 0，58 轮，GPU 0/1 全局 batch 28，640 验证、480–800 多尺度，
warmup 432、flat 29、no_aug 8，policy [4,29,50]，mixup [4,29]，stop 50，AdamW 主 LR 3.5e-4 / 骨干 3.5e-5，
EMA 0.9999，AMP FP16 + FP32 loss/retry，关闭 ZoomOut/IoUCrop。按 val AP 选权重，报告 test。

判定（58 轮，对照 baseline test 46.18、REM test 47.10）：

- test ≥ 47.7（相对 baseline ≥ +1.5）：作为主模块，跑 s8+s16 消融，并在 102 轮确认。
- 47.1–47.7：比 REM 好但未达标，结合诊断决定下一步改法。
- < 47.1：多层级无益，主模块保持 stride-8 REM，改进转向诊断指出的方向。

## 4. 服务器命令（旧服务器）

更新代码：

```bash
cd /home/zxy/sar/repos/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
```

工作区有改动或分叉时先查看，不要 reset。数据不需要重新准备。

### 第一步：test 诊断（约几分钟，两卡）

```bash
export BASELINE=/data2/zxy/sar/experiments/imported_baseline102_LMw1Yy
export REM=/data2/zxy/sar/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_rem_102e_b28_old_server_seed0_20261101_145427_rtK11d
nvidia-smi -i 0,1
SPLIT=test bash tools/run_rem_102e_diagnosis.sh "$BASELINE" "$REM"
```

把打印出的 `OUT=` 设为 `DIAG`，查看进度：

```bash
export DIAG=/data2/zxy/sar/experiments/diagnostics/替换为rem102_comparison_test目录
tail -f "$DIAG/console.log"
ps -p "$(cat "$DIAG/launcher.pid")" -o pid,etime,args
nvidia-smi -i 0,1
```

出现 `$DIAG/DONE` 后，再跑一次 val（新目录）：

```bash
ls "$DIAG/DONE" && SPLIT=val bash tools/run_rem_102e_diagnosis.sh "$BASELINE" "$REM"
```

结果：`$DIAG/report/report.md`（类别无关 AP、各类 AP、定位与型号表）和 `$DIAG/report/confusion_*.csv`。
把 test 与 val 两份 `report.md` 贴回即可。中断后恢复：

```bash
bash tools/run_rem_102e_diagnosis.sh --resume "$DIAG"
```

### 第二步：REM-Pyramid 训练（诊断结束、两卡空闲后）

```bash
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_pyramid_58e_b28_old_server.yml
```

启动器检查两卡空闲，新建目录并保存解析配置、Git revision、seed、batch、协议与初始化路径，
以 `nohup ... > console.log 2>&1 < /dev/null &` 启动并保存 PID 后 `disown`，打印 `STARTED OUT=...`。

监控（重连后重新设置 OUT 并激活环境）：

```bash
export OUT=/home/zxy/sar/experiments/替换为启动器打印的目录
tail -f "$OUT/console.log"
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
nvidia-smi -i 0,1
tail -n 3 "$OUT/log.txt"
python - "$OUT/log.txt" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
for r in rows[-3:]:
    v = r['test_coco_eval_bbox']
    print(r['epoch'], 'val AP', round(v[0] * 100, 2), 'AP75', round(v[2] * 100, 2),
          'alpha s8/s16/s32', [round(r.get(k, float('nan')), 4) for k in ('train_rem_alpha', 'train_rem_p4_alpha', 'train_rem_p5_alpha')])
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

只在进程已停止时恢复：

```bash
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

`last.pth` 只在 epoch < 50 时更新，不是最终权重。

训练完成后按 val 选权重并评估 test（批量工具会跳过已评估的运行）：

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

## 5. 本地验证

Windows Conda `pytorch` 环境（CPU）：`tests/test_rem_pyramid.py` 检查默认 REM 的 state_dict 键不变、
Pyramid 初始为恒等且 stride-8 REM 初始化与单 REM 相同、三层都被调用并有梯度、alpha 日志键、非法层级报错、
两个配置相对 REM 58 轮只改 `rem_feat_indices`；`tests/test_rem_102e_diagnosis.py` 新增 test 划分、
类别无关缺口与型号混淆的合成数据检查。未在服务器执行，没有训练或诊断结果。
