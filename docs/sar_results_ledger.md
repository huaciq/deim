# SAR 实验结果总表

本文是论文实验的统一结果台账。第 1 节汇总各训练日志最佳 val 轮次的指标
（用户整理的 `C:\project\实验记录\实验记录表.xlsx`）和对应权重的 test 独立评估。
生成的评估文件留在服务器实验目录，不提交 Git。

## 1. val 与 test 结果

SAR-AIRcraft-1.0，DEIM-D-FINE-M（COCO M 初始化），7 类，train 3047 / val 442 / test 879，
seed 0，全局 batch 28，旧服务器。val 为训练日志最佳轮次；test 为该轮对应的
`best_stg1` / `best_stg2` 在 test.json 上的独立评估（`tools/run_sar_test_evals.py`，
2026-11-02 批次，用户下载的 `C:\project\实验记录\summary.md`）。test 不参与选择权重。
Δ 只在同一协议内对 baseline 作差；AP75/APm/APl 列为 test 上相对 baseline 的差值。

### 58 轮协议（baseline val 63.08 / test 46.18）

| 方法 | val AP | Δval | test AP | Δtest | ΔAP75 | ΔAPm | ΔAPl | 选中 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 63.08 | — | 46.18 | — | — | — | — | stg1/44 |
| CLCR-Lite | 63.79 | +0.71 | **47.42** | **+1.24** | +2.56 | +1.23 | −1.26 | stg1/44 |
| REM | **64.22** | **+1.14** | 47.10 | +0.92 | +0.77 | +1.39 | +0.88 | stg1/46 |
| W-FDR | 63.91 | +0.83 | 46.83 | +0.65 | +0.76 | +1.68 | −3.91 | stg2/57 |
| OST-DA v2 | 63.25 | +0.17 | 46.82 | +0.64 | −0.40 | +0.72 | +0.97 | stg1/49 |
| PGD-REM | 63.30 | +0.22 | 46.79 | +0.61 | +0.74 | +1.06 | −5.43 | stg2/51 |
| REM+PACFG | 63.65 | +0.57 | 46.30 | +0.12 | +0.78 | +0.46 | −3.01 | stg1/47 |
| CLCR | 63.70 | +0.62 | 46.27 | +0.09 | +0.57 | +1.00 | −0.84 | stg1/48 |
| CrossGate-DBDM | 62.85 | −0.23 | 46.16 | −0.02 | +0.07 | +0.31 | +1.45 | stg1/48 |
| REM-Pyramid（s8/16/32） | 62.92 | −0.16 | 46.29 | +0.11 | −1.58 | +0.86 | +0.31 | stg1/42 |
| PAFP | 63.45 | +0.37 | 46.13 | −0.05 | −0.14 | +0.11 | +0.11 | stg1/44 |
| OST-DA v1 | 63.07 | −0.01 | 45.98 | −0.20 | −0.78 | +0.99 | −4.66 | stg1/46 |
| REM+CLCR-Lite | 63.89 | +0.81 | 45.98 | −0.20 | −0.50 | +0.23 | −2.63 | stg1/42 |
| REM+CLCR-Lite detach30 | 62.98 | −0.10 | 45.96 | −0.22 | +0.09 | −0.35 | +2.16 | stg1/44 |
| REM point-strip | 62.55 | −0.53 | 45.84 | −0.34 | −0.78 | +0.78 | +0.03 | stg1/49 |
| REM+W-FDR | 63.57 | +0.49 | 45.71 | −0.47 | −0.85 | +0.46 | +1.23 | stg2/53 |
| REM-s8 + PACFG-s16 | 62.59 | −0.49 | 45.59 | −0.59 | +2.24 | −0.80 | −0.17 | stg1/48 |
| SPI | 62.50 | −0.58 | 45.51 | −0.67 | −0.27 | +0.75 | −7.19 | stg1/47 |
| REM square | 62.34 | −0.74 | 45.11 | −1.07 | −1.49 | −0.87 | +0.30 | stg1/43 |
| REM space-to-depth | 62.43 | −0.65 | 44.25 | −1.93 | −3.44 | −0.13 | −8.97 | stg1/47 |
| SDCP | 62.21 | −0.87 | 44.17 | −2.01 | −1.52 | −1.19 | −4.69 | stg1/46 |
| PACFG | 63.86 | +0.78 | 43.32 | **−2.86** | −2.45 | −1.86 | −2.65 | stg1/48 |

baseline test：AP50 67.27，AP75 51.52，APm 45.40，APl 50.72，AR100 68.15。

### 102 轮协议（baseline val 64.51 / test 45.74）

| 方法 | val AP | Δval | test AP | Δtest | ΔAP75 | ΔAPm | ΔAPl | 选中 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 64.51 | — | 45.74 | — | — | — | — | stg2/90 |
| REM | **65.22** | **+0.71** | **47.31** | **+1.57** | +1.63 | −0.46 | +5.92 | stg2/98 |
| CLCR-Lite | 64.54 | +0.03 | 46.43 | +0.69 | +0.34 | +0.57 | +2.03 | stg1/75 |
| W-FDR | 63.76 | −0.75 | 44.21 | −1.53 | −2.24 | −1.19 | −0.78 | stg1/88 |
| BCCQ | 61.80 | −2.71 | 42.82 | −2.92 | −3.57 | −4.25 | +2.15 | stg1/81 |

baseline test：AP50 65.37，AP75 51.42，APm 46.82，APl 43.11，AR100 68.92。

102 轮 SCS、DSG、SLQC、PACFG、MDCC 只有 val 记录（分别为 64.0、64.2、63.4、63.5、64.0，
均低于 baseline），不在旧服务器批次中，仍缺 test。

### OGSOD（DEIM-D-FINE-S，120 轮，仅 val）

| 方法 | batch | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32 | 57.9 | 91.4 | 56.1 | 50.9 | 85.1 | 85.4 |
| BCM | 32 | 58.3 | 91.8 | 57.0 | 51.5 | 85.0 | 85.4 |
| baseline | 24 | 59.3 | 92.4 | 58.0 | 52.9 | 85.8 | 86.5 |
| SPI | 24 | 59.4 | 92.4 | 58.0 | 52.6 | 82.6 | 87.1 |
| SBR-v2 | 28 | 58.6 | 91.8 | 57.2 | 52.2 | 85.3 | 86.7 |

### 结论

> **2026-11-03 重要更正**：val 82%、test 50% 的目标与 train 中的目标近乎相同（重叠切片），
> 见过的目标几乎全对，没见过的目标在 val/test 上表现相近。下面关于 val–test 差距和模块增益的结论需要在
> "未见实例"协议下复核，REM 在未见目标上的点统计反而低于 baseline。详见
> [型号混淆线索诊断](sar_confusion_cues_diagnosis.md) 的 SAR 特性检查结果。

1. **REM 是唯一在两个协议、两个划分上都为正的模块**：58 轮 val +1.14 / test +0.92，
   102 轮 val +0.71 / test +1.57；102 轮 test AP75 +1.63。
2. **CLCR-Lite 是第二个候选**：test 上 58 轮 +1.24（全部方法最高）、102 轮 +0.69；
   但 102 轮 val 只有 +0.03，val 低估了它。
3. **PACFG 是 val 误导的典型**：58 轮 val +0.78，test −2.86。W-FDR 58 轮两边都为正，102 轮两边都为负。
4. **组合没有叠加收益**：REM+CLCR-Lite test −0.20，而两个单模块分别为 +0.92 和 +1.24；
   REM+W-FDR −0.47，REM+PACFG +0.12，全部低于 REM 单模块。
5. **REM 机制消融支持原设计**：把 Haar 换成 space-to-depth（−1.93）、方向条带换成方形卷积（−1.07）、
   改为 point-strip（−0.34），test 都低于原 REM（+0.92）。这一组可直接作为论文的组件消融。
6. **val–test 差距是系统性的**：58 轮 21 组 val−test 平均 17.3 AP（标准差 0.9），102 轮 5 组平均 18.7（标准差 0.6），
   AP50 也从约 88 降到约 67。所有运行同样下降，不是个别运行的错误，而是 test 集本身更难或分布不同。
   58 轮 val 与 test 排序的 Spearman 相关 0.63（Pearson 0.47）：val 能分出大差距，分不清 1 AP 以内的差别。
7. **58 轮 baseline 的 test 高于 102 轮**（46.18 对 45.74），58 轮协议的验收条件（test ≥ 45.2）成立。
8. **APs 不可用，APl 噪声很大**：val 只有 1 个 small GT；APl 在各组之间波动可达 ±9，
   不应单独作为结论。
9. 全部为单种子。同一方法在两协议间的 test 差值波动约 1–2 AP（如 W-FDR +0.65→−1.53），
   因此 1 AP 左右的差异需要多种子确认。

## 2. 汇总脚本口径

### 结果总表 `tools/collect_sar_results.py`

只读扫描实验根目录，凡同时含 `log.txt` 和 `resolved_config.yml` 的目录记为一次运行。

- **方法名**：由解析配置中 HGNetv2 / HybridEncoder / DFINETransformer / DEIMCriterion 的
  `use_*: True` 开关自动拼接（REM 带上 variant 和 residual mode），全部关闭记为 `baseline`。
- **协议**：数据集、轮数、warmup、flat、no_aug、stop_epoch、全局 batch、验证尺寸、多尺度、
  增强 policy、主 LR 全部相同才算同一协议；表格按"数据集 · 协议"分组。
- **val**：`log.txt` 中 `test_coco_eval_bbox` 最佳 AP 的那一轮（索引 0..5 为 AP/AP50/AP75/APs/APm/APl，
  8 为 AR100，乘 100）。按 solver 规则，该轮 ≥ stop_epoch 时对应 `best_stg2`，否则对应 `best_stg1`。
- **test**：运行目录下任意 `eval.pth`（`--test-only` 输出）。检查点名取同目录 `console.log` 中的
  `Resume checkpoint from`；划分按评估图像数判定（442=val，879=test），否则看日志中的标注文件名。
- **Δ**：减去同协议、同 seed、已完成 baseline 的均值；没有对应 baseline 时留空。

### 批量 test 评估 `tools/run_sar_test_evals.py`

- 自动找出指定根目录下所有已跑完的 SAR-AIRcraft 运行，也可用 `--run` 逐个指定。
- 每个运行按上面的 val 规则选 `best_stg1` / `best_stg2`，test AP 不参与选择。
- 依次在 GPU 0、1 上用该运行自己的 `resolved_config.yml` 执行 `train.py -r ... --test-only`，
  只把验证图像目录和标注文件改为本机 test 集，所以从新服务器转来的运行也能直接评估。
- 输出到 `RUN/test_eval_<stage>_<批次目录名>/`（`eval.pth`、`console.log`、`command.json`），不覆盖已有文件；
  该阶段已有 test 评估时跳过。中断后用同一条命令重跑即可续上。
- 批次目录中 `plan.tsv` 为评估计划，`summary.md` / `summary.tsv` / `summary.json`（含各类 AP）每完成一组更新一次。
- 评估图片数不是 879 时，状态会标出来。

## 3. 服务器命令（旧服务器）

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

工作区有改动或与远端分叉时先查看差异，不要 reset。数据已经准备好，不需要转换。

### 批量 test 评估

先只生成计划，核对要评估的运行（不占 GPU）：

```bash
export TESTBATCH=/home/zxy/sar/experiments/test_evals_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
python tools/run_sar_test_evals.py --plan-only \
  --root /home/zxy/sar/experiments --root /data2/zxy/sar/experiments \
  --data-root /home/zxy/sar/datasets/SAR-AIRcraft-1.0 --output "$TESTBATCH"
column -t -s $'\t' "$TESTBATCH/plan.tsv" | cut -c1-200
```

`status=planned` 的行会被评估；`incomplete` 和 `missing best_stg*.pth` 的行跳过。
同一配置有多次运行（例如重启过的旧目录）时，改用 `--run` 只列出要用的目录。
确认 GPU 0、1 空闲后，在后台启动：

```bash
nvidia-smi -i 0,1
ulimit -Sn 65535
nohup python tools/run_sar_test_evals.py \
  --root /home/zxy/sar/experiments --root /data2/zxy/sar/experiments \
  --data-root /home/zxy/sar/datasets/SAR-AIRcraft-1.0 --output "$TESTBATCH" \
  > "$TESTBATCH/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$TESTBATCH/launcher.pid"
disown "$PID"
tail -f "$TESTBATCH/console.log"
```

只评估部分运行时，把两个 `--root` 换成若干个 `--run /路径/运行目录`。

重连后查看进度、GPU 和结果：

```bash
export TESTBATCH=/home/zxy/sar/experiments/替换为批次目录
ps -p "$(cat "$TESTBATCH/launcher.pid")" -o pid,etime,args
nvidia-smi -i 0,1
tail -n 20 "$TESTBATCH/console.log"
cat "$TESTBATCH/summary.md"
```

查看某一组正在评估的日志（目录名见批次 console.log 中的 `EVAL ... ->` 行）：

```bash
tail -f /替换为EVAL目录/console.log
grep -A2 "IoU=0.50:0.95 | area=   all | maxDets=100 ]" /替换为EVAL目录/console.log
```

中断后恢复：用同一 `TESTBATCH` 重新执行上面的 nohup 启动命令，已完成的组会被跳过。
全部完成后批次 console.log 最后一行为 `ALL FINISHED`，把 `summary.md` 贴回即可填入第 1 节。

如果部分运行（例如 102 轮 SCS / DSG / SLQC）只在新服务器上，需在新服务器的 `/root/zxy/deim`
上运行同一工具，路径换为 `--root /root/zxy/experiments --data-root /root/zxy/datasets/SAR-AIRcraft-1.0`，
批次目录放在 `/root/zxy/experiments/` 下。

### 结果总表（可选）

```bash
LEDGER=/home/zxy/sar/experiments/results_ledger_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
python tools/collect_sar_results.py \
  --root /home/zxy/sar/experiments --root /data2/zxy/sar/experiments \
  --output "$LEDGER" 2> "$LEDGER.stderr.log"
cat "$LEDGER/ledger.md"
```

开发机目前没有 Python 环境，`tests/test_collect_sar_results.py` 未在本地执行。
它只用 CPU 和合成目录，可在服务器上运行：

```bash
CUDA_VISIBLE_DEVICES= python -m unittest tests.test_collect_sar_results -v
```

查看单个运行的训练日志、逐轮指标和 TensorBoard：

```bash
tail -f "$OUT/console.log"
tail -n 3 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

这两个工具都不启动、恢复或修改训练；训练和恢复命令见各实验文档
（如 [58 轮旧服务器交接](sar_aircraft_58e_old_server.md)）。
