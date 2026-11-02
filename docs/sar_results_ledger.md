# SAR 实验结果总表

本文是论文实验的统一结果台账。数字以服务器上 `tools/collect_sar_results.py`
从各运行目录直接汇总的结果为准；第 1 节中仓库已有文档的数字，是汇总前的参照，
需要用汇总结果核实后替换。生成的 CSV/JSON/Markdown 留在服务器实验目录，不提交 Git；
核实后的表格手工整理进本文。

## 1. 仓库已有记录（待核实）

SAR-AIRcraft-1.0，DEIM-D-FINE-M，7 类，train 3047 / val 442 / test 879，seed 0，
全局 batch 28。val AP 为最佳轮次的逐轮验证值。不同协议之间不作差。

| 协议 | 方法 | val AP | 其它指标 | test AP | 来源 |
|---|---|---:|---|---:|---|
| 102e | baseline | 64.51（epoch 90） | AP75 72.9，APm 64.4，APl 57.9 | 45.7 | [MPQ](sar_mpq_experiments.md)、[58 轮协议](sar_aircraft_deim_m_58e_protocol.md) |
| 102e | REM | 65.22（配对复算 64.51→65.22） | AP50 88.6，AP75 74.1，APm 64.0，APl 58.5；A330 +5.14，A220 −1.97 | — | [REM 102 轮诊断](sar_rem_102e_diagnosis.md)、[机制诊断](sar_rem_mechanism_probe.md) |
| 102e | MPQ-single | 64.36 | test 未显示收益 | — | [BCCQ 研究](sar_band_conditioned_class_queries_research_2026-11-01.md) |
| 58e | baseline | ≈63.1 | — | — | [REM per-band](sar_rem_per_band_experiment.md) |
| 58e | REM | ≈64.2 | — | — | [REM 机制消融](sar_rem_mechanism_ablations.md) |
| 58e | REM+W-FDR | ≤ REM | — | — | 同上 |
| 58e | REM、PACFG 单模块 | — | AP_S 均低于 baseline | — | [REM+PACFG](sar_rem_pacfg_experiment.md) |

问题记录：

- test AP 45.7 和"test 曲线在 epoch 59 左右进入平台"引自 `docs/research_diagnosis_2026-10-28.md`，
  该文件不在仓库中。val 与 test 相差约 19 AP，需要用汇总出的独立 test 评估复核。
- 58 轮协议的验收条件（test AP ≥ 45.2）没有记录结果。
- 以上全部为单种子结果。

已实现但仓库中没有结果记录的配置：CLCR、CLCR-Lite（58/102e、detach30）、SDCP、PGD-REM 系列、
PAFP、PACFG（58/102e）、CrossGate-DBDM、OST-DA v1/v2、SPI 系列、W-FDR（102e）、
REM 变体（per-band、point-strip、square、space-to-depth、s8+PACFG-s16）、
uniform4、sampling464、SCS-M1、BCCQ、MDCC、REM+MDCC、`sar_aircraft_detail`、`sar_aircraft_modules`、`sar_aircraft_mpq`。

## 2. 汇总脚本口径

`tools/collect_sar_results.py` 只读扫描实验根目录，凡同时含 `log.txt` 和 `resolved_config.yml`
的目录记为一次运行。

- **方法名**：由解析配置中 HGNetv2 / HybridEncoder / DFINETransformer / DEIMCriterion 的
  `use_*: True` 开关自动拼接（REM 带上 variant 和 residual mode），全部关闭记为 `baseline`。
- **协议**：数据集、轮数、warmup、flat、no_aug、stop_epoch、全局 batch、验证尺寸、多尺度、
  增强 policy、主 LR 全部相同才算同一协议；表格按"数据集 · 协议"分组。
- **val**：`log.txt` 中 `test_coco_eval_bbox` 最佳 AP 的那一轮（索引 0..5 为 AP/AP50/AP75/APs/APm/APl，
  8 为 AR100，乘 100）。按 solver 规则，该轮 ≥ stop_epoch 时对应 `best_stg2`，否则对应 `best_stg1`。
- **test**：运行目录下任意 `eval.pth`（`--test-only` 输出）。检查点名取同目录 `console.log` 中的
  `Resume checkpoint from`；划分按评估图像数判定（442=val，879=test），否则看日志中的标注文件名。
  主表中的 test 列只取所选检查点的评估，其它检查点（如曲线 checkpoint）列在附表里。
- **Δ**：减去同协议、同 seed、已完成 baseline 的均值；没有对应 baseline 时留空。
- **状态**：最后一轮达到 `epoches-1` 为 complete；`launcher.pid` 进程仍存在为 running；否则为 stopped。

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

工作区有改动或与远端分叉时先查看差异，不要 reset。汇总只需 CPU，不占 GPU，不需要准备数据。

开发机本次没有可用的 Python 环境，定向测试 `tests/test_collect_sar_results.py` 未在本地执行。
它只用 CPU 和合成目录，检查选权重、划分识别、Δ 计算和 COCO 指标。首次使用前在服务器上运行：

```bash
CUDA_VISIBLE_DEVICES= python -m unittest tests.test_collect_sar_results -v
```

生成总表（两个实验根目录都扫描，输出到新目录）：

```bash
LEDGER=/home/zxy/sar/experiments/results_ledger_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
python tools/collect_sar_results.py \
  --root /home/zxy/sar/experiments \
  --root /data2/zxy/sar/experiments \
  --output "$LEDGER" 2> "$LEDGER.stderr.log"
cat "$LEDGER.stderr.log"
cat "$LEDGER/ledger.md"
```

输出文件：`ledger.md`（分组主表、全部独立评估、缺 test 评估的已完成运行），
`runs.csv`、`test_evaluations.csv`、`ledger.json`（含各类 AP）。把 `ledger.md` 的内容贴回，
用于替换第 1 节。

### 补跑缺失的 test 评估

`ledger.md` 末尾会列出已完成、但所选检查点还没有 test 评估的运行。对每个这样的运行执行
（`OUT` 和 `STAGE` 取自该列表；两卡；不按 test AP 重新选权重）：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为运行目录
STAGE=best_stg2   # 按 ledger.md 中的 selected 填写
EVAL=$(mktemp -d "$OUT/test_eval_${STAGE}_XXXXXX")
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/$STAGE.pth" --test-only --seed 0 --use-amp \
  --output-dir "$EVAL" --summary-dir "$EVAL/summary" \
  -u val_dataloader.dataset.ann_file=/home/zxy/sar/datasets/SAR-AIRcraft-1.0/COCO/test.json \
  > "$EVAL/console.log" 2>&1 < /dev/null &
EVAL_PID=$!
printf '%s\n' "$EVAL_PID" > "$EVAL/launcher.pid"
disown "$EVAL_PID"
tail -f "$EVAL/console.log"
```

查看进度和 GPU：

```bash
ps -p "$(cat "$EVAL/launcher.pid")" -o pid,etime,args
nvidia-smi -i 0,1
grep -A2 "IoU=0.50:0.95 | area=   all | maxDets=100 ]" "$EVAL/console.log"
```

评估写入 `$EVAL/eval.pth`，再次运行汇总命令即可进入主表。训练中的运行可随时汇总；
其行显示 running，最佳 val 为截至当前的值。查看某一运行的训练日志、逐轮指标和 TensorBoard：

```bash
tail -f "$OUT/console.log"
tail -n 3 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

本工具不启动、恢复或修改训练；训练和恢复命令见各实验文档
（如 [58 轮旧服务器交接](sar_aircraft_58e_old_server.md)）。
