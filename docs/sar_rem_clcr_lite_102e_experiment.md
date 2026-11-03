# SAR-AIRcraft：REM + CLCR-Lite，102 轮组合实验

日期：2026-11-02。路线依据见 [主模块研判](sar_rem_pyramid_experiment.md)，结果台账见 [结果总表](sar_results_ledger.md)。

## 为什么做这一组

- REM-Pyramid（stride 8/16/32）58 轮 test 46.29，比 baseline 46.18 高 0.11，比单层 REM 47.10 低 0.81；
  val 62.92 也低于 REM 64.22 和 baseline 63.08。多层级无益，主模块保持 stride-8 REM。
- REM 的收益随训练变长而变大：test +0.92（58 轮）→ +1.57（102 轮），102 轮已进入主模块 +1.5～2.0 的目标区间。
  后续主实验与组合都在 102 轮协议下做。
- CLCR-Lite 是唯一在 test 上有收益的分类侧模块（58 轮 +1.24、102 轮 +0.69），也是诊断指出的窄体机型号判错的直接对象。
  REM+CLCR-Lite 只有 58 轮单次结果（test 45.98），尚未在 102 轮检验。

本组回答：在 102 轮 REM 之上，CLCR-Lite 能否再加 1.0 以上。对照为 102 轮 REM（test 47.31，val 65.22）。

## 结果（2026-11-03）：不能叠加，停止 CLCR-Lite

运行 `/data2/zxy/sar/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_rem_clcr_lite_102e_b28_old_server_seed0_20261103_102028_MOSNxK`，
最佳 val 在 epoch 82（best_stg1）；test 由 `tools/run_sar_test_evals.py` 评估（879 张）。

| 102 轮 | val AP | test AP | AP50 | AP75 | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 64.51 | 45.74 | 65.37 | 51.42 | 46.82 | 43.11 |
| REM | 65.22 | **47.31** | 67.17 | 53.05 | 46.36 | 49.03 |
| CLCR-Lite | 64.54 | 46.43 | 67.06 | 51.76 | 47.39 | 45.14 |
| REM+CLCR-Lite | 64.67 | 45.42 | 64.51 | 49.88 | 46.12 | 45.24 |

按判定（< 47.31）停止 CLCR-Lite：58 轮（45.98 对 REM 47.10）和 102 轮两次组合都低于 REM 单模块，
102 轮还低于 baseline。

test 各类 AP 与 REM 的差（REM 取 102 轮配对诊断导出，每卡 batch 8，与 `--test-only` 口径有约 0.3 AP 的差别）：

| 类别 | test GT | REM | REM+CLCR-Lite | 差 |
|---|---:|---:|---:|---:|
| A220 | 460 | 41.24 | 43.35 | +2.11 |
| A320/321 | 52 | 66.84 | 55.33 | −11.51 |
| A330 | 31 | 82.87 | 82.75 | −0.12 |
| ARJ21 | 362 | 36.46 | 30.26 | −6.20 |
| Boeing737 | 550 | 23.02 | 24.40 | +1.38 |
| Boeing787 | 454 | 41.28 | 39.77 | −1.51 |
| other | 1041 | 41.81 | 42.09 | +0.28 |

A320/321 一类就让 7 类平均下降约 1.6。按 [重复目标检查](sar_confusion_cues_diagnosis.md)，test 中没见过的飞机里
没有 A320/321 与 A330，这两类的 83 个 GT 全部是 train 中出现过的飞机的重叠切片；REM 的 +1.57 里约 +1.44 也来自这两类。
单次运行中这两类的大幅波动反映的是对见过目标的记忆差异，1 AP 量级的模块比较在这个测试集上需要多种子才能判定。
后续路线按 [协议调研](sar_aircraft_protocol_survey.md) 第 4 节，先确定输入尺寸等基准设置。

## 配置

`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_clcr_lite_102e_b28_old_server.yml`：
继承旧服务器 CLCR-Lite 102 轮配置，只加 `HybridEncoder.use_rem: True`（stride-8，`rem_feat_indices` 默认 [0]）。
CLCR-Lite 为原版：8 维输入、Linear(8,16)→SiLU→Linear(16,1)，四层 decoder 共享 162 个参数，输出层零初始化，
不 detach，从 epoch 0 启用。`tests/test_rem_clcr_lite_102e.py` 检查它相对 REM 102 轮只多 CLCR-Lite，
相对 CLCR-Lite 102 轮只多 REM。

协议与冻结 102 轮 baseline 相同：DEIM-D-FINE-M，COCO M 权重 `-t`，seed 0，GPU 0/1 全局 batch 28，
640 验证、480–800 多尺度，warmup 2000、flat 49、no_aug 12，policy [4,49,90]，mixup [4,49]，stop 90，
AdamW 主 LR 3.5e-4 / 骨干非 BN 3.5e-5，WD 1e-4，EMA 0.9999，AMP FP16 + FP32 loss/retry，关闭 ZoomOut/IoUCrop。
按 val AP 选权重，报告 test。

判定（test，对照 REM 47.31、baseline 45.74）：

- ≥ 48.3（比 REM 高 ≥ 1.0）：CLCR-Lite 作为第二模块保留。
- 47.31–48.3：有叠加但未达标，看 Boeing737 / A220 / ARJ21 各类 AP 再决定是否改 CLCR-Lite。
- < 47.31：不能叠加，停掉 CLCR-Lite，第二模块改走特征一侧。

## 服务器命令（旧服务器）

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

启动（启动器检查两卡空闲、新建目录、保存解析配置 / Git revision / seed / batch / 协议 / 初始化路径，
以 `nohup ... > console.log 2>&1 < /dev/null &` 启动，保存 PID 后 `disown`，打印 `STARTED OUT=...`）：

```bash
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
nvidia-smi -i 0,1
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_clcr_lite_102e_b28_old_server.yml
```

重连后设置运行目录并监控：

```bash
export OUT=$(ls -dt /home/zxy/sar/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_rem_clcr_lite_102e_b28_old_server_seed0_* | head -1)
echo "$OUT"
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
          'rem_alpha', round(r.get('train_rem_alpha', float('nan')), 4))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

只在进程已停止时恢复（`last.pth` 只在 epoch < 90 时更新，不是最终权重）：

```bash
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

训练完成（`log.txt` 最后一轮为 101）后，按 val 选权重评估 test：

```bash
export TESTBATCH=/home/zxy/sar/experiments/test_evals_rem_clcr_lite_102e
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

`summary.json` 含各类 AP。需要看型号混淆时，用配对诊断与 102 轮 REM 比较：

```bash
export REM=/data2/zxy/sar/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_rem_102e_b28_old_server_seed0_20261101_145427_rtK11d
SPLIT=test bash tools/run_rem_102e_diagnosis.sh "$REM" "$OUT"
```

该诊断中 "baseline" 一栏是 REM，"REM" 一栏是 REM+CLCR-Lite。
