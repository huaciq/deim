# SAR 实验结果总表

本文是论文实验的统一结果台账。第 1 节是用户从各训练日志最佳 val 轮次整理的指标
（原表 `C:\project\实验记录\实验记录表.xlsx`，2026-11-02 读取），尚无 test 结果；
第 3 节的批量评估完成后补入 test 列。生成的评估文件留在服务器实验目录，不提交 Git。

## 1. 已记录的 val 结果

SAR-AIRcraft-1.0，DEIM-D-FINE-M（COCO M 初始化），7 类，train 3047 / val 442 / test 879，
seed 0，全局 batch 28。数值为训练日志中最佳 val 轮次的 COCO 指标，原表保留三位小数
（0..1），此处乘 100，分辨率 0.1 AP。Δ 只在同一协议内对 baseline 作差。

### 58 轮协议（baseline 63.1）

| 方法 | AP | ΔAP | AP50 | AP75 | APm | APl | test AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 63.1 | — | 87.4 | 71.3 | 61.8 | 58.4 | 待测 |
| REM | **64.2** | **+1.1** | 88.6 | 71.7 | 63.1 | 58.2 | 待测 |
| PACFG | 63.9 | +0.8 | 87.9 | 73.6 | 62.9 | 55.6 | 待测 |
| W-FDR | 63.9 | +0.8 | 88.1 | 71.1 | 62.6 | 58.4 | 待测 |
| REM+CLCR-Lite | 63.9 | +0.8 | 88.7 | 71.9 | 62.5 | 58.0 | 待测 |
| CLCR-Lite | 63.8 | +0.7 | 88.5 | 72.9 | 62.7 | 56.4 | 待测 |
| CLCR | 63.7 | +0.6 | 88.3 | 73.3 | 63.2 | 55.8 | 待测 |
| REM+PACFG | 63.6 | +0.5 | 88.2 | 72.8 | 63.0 | 56.8 | 待测 |
| REM+W-FDR | 63.6 | +0.5 | 88.3 | 69.6 | 62.3 | 56.2 | 待测 |
| PAFP | 63.4 | +0.3 | 88.9 | 71.2 | 62.7 | 57.1 | 待测 |
| PGD-REM | 63.3 | +0.2 | 87.4 | 71.4 | 62.8 | 56.0 | 待测 |
| OST-DA | 63.1 | 0.0 | 88.0 | 72.8 | 62.3 | 56.2 | 待测 |
| CrossGate-DBDM | 62.9 | −0.2 | 86.7 | 71.8 | 62.9 | 55.2 | 待测 |
| SPI | 62.5 | −0.6 | 86.8 | 71.0 | 62.6 | 53.6 | 待测 |
| SDCP | 62.2 | −0.9 | 87.0 | 73.1 | 61.4 | 55.4 | 待测 |

### 102 轮协议（baseline 64.5）

| 方法 | AP | ΔAP | AP50 | AP75 | APm | APl | test AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 64.5 | — | 88.1 | 72.9 | 64.4 | 57.9 | 待测（旧记录 45.7，来源文档缺失） |
| REM | **65.2** | **+0.7** | 88.6 | 74.1 | 64.0 | 58.5 | 待测 |
| CLCR-Lite | 64.5 | 0.0 | 88.7 | 73.0 | 63.7 | 56.7 | 待测 |
| MPQ-single | 64.36 | −0.15 | — | — | — | — | [MPQ](sar_mpq_experiments.md) 记录 |
| DSG | 64.2 | −0.3 | 87.8 | 72.8 | 63.4 | 57.0 | 待测 |
| SCS | 64.0 | −0.5 | 87.4 | 71.2 | 62.8 | 57.1 | 待测 |
| MDCC | 64.0 | −0.5 | 88.2 | 72.0 | 63.7 | 56.9 | 待测 |
| W-FDR | 63.8 | −0.7 | 86.9 | 72.0 | 63.4 | 56.6 | 待测 |
| PACFG | 63.5 | −1.0 | 86.9 | 72.0 | 62.8 | 56.5 | 待测 |
| SLQC | 63.4 | −1.1 | 87.1 | 71.7 | 63.5 | 56.4 | 待测 |
| BCCQ | 61.8 | −2.7 | 84.5 | 69.8 | 60.3 | 54.5 | 待测 |

### OGSOD（DEIM-D-FINE-S，120 轮）

| 方法 | batch | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32 | 57.9 | 91.4 | 56.1 | 50.9 | 85.1 | 85.4 |
| BCM | 32 | 58.3 | 91.8 | 57.0 | 51.5 | 85.0 | 85.4 |
| baseline | 24 | 59.3 | 92.4 | 58.0 | 52.9 | 85.8 | 86.5 |
| SPI | 24 | 59.4 | 92.4 | 58.0 | 52.6 | 82.6 | 87.1 |
| SBR-v2 | 28 | 58.6 | 91.8 | 57.2 | 52.2 | 85.3 | 86.7 |

原表中 SPI（b24）出现两次且数值相同，按一次计。

### 读表结论

1. **只有 REM 在两个协议下都超过 baseline**（58 轮 +1.1，102 轮 +0.7），102 轮 AP75 +1.2。
2. **58 轮的小幅收益在 102 轮没有保持**：PACFG +0.8→−1.0，W-FDR +0.8→−0.7，CLCR-Lite +0.7→0.0。
   因此 58 轮下 ≤0.8 AP 的差异不能作为筛选依据；OGSOD 上仅 batch 32→24 就让 baseline 变化 1.4 AP，
   也说明单种子差异的量级接近 1 AP。
3. **所有与 REM 的组合都低于 REM 单模块**（REM+PACFG 63.6、REM+W-FDR 63.6、REM+CLCR-Lite 63.9，均 <64.2）。
4. **APs 不可用**：val 原图 COCO small 只有 1 个 GT（见 [S2 细节实验](sar_s2_detail_experiments.md)），
   各组 APs 呈 0.1/0.5/0.6/0.7 这类离散值，表中不列出，论文中也不应作为依据。
5. 全部为单种子，尚无 test 结果；论文报告 test，方法选择保持按 val。

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
