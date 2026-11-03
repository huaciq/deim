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
