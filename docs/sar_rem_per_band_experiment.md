# SAR-AIRcraft：原 REM 长程验证与四子带残差解耦

## 本轮问题与实验顺序

本轮先检验原 REM 在 102 轮下能否保留相对 baseline 的收益，再检验共享残差系数
是否限制了 REM。已有 58 轮 baseline 约 63.1 AP、REM 约 64.2 AP；已有 102 轮
baseline 约 64.5 AP。不同训练协议分别比较，不将 58 轮模块直接减去 102 轮 baseline。

1. 原 REM / 102 轮：首个正式运行，回答长程收益问题。
2. REM-per-band / 58 轮：与已报告的原 REM / 58 轮比较，隔离残差系数的作用。
3. REM-per-band / 102 轮：配置已完整提供；短程改进成立后再运行，与原 REM / 102 轮比较。

这三组都从同一 COCO M 权重以 `-t` 新训，不从已有 58 轮 REM 权重继续训练。
本轮不加入 CLCR、W-FDR、PACFG、SPI 或 OST-DA，也不改变匹配、损失及后处理。
主模块 +1.5～2 AP 是筛选目标，不是本次改动的已知结果。

## 数学定义与插入位置

插入位置仍是 HybridEncoder 的 stride-8 input_proj 之后、FPN/PAN 融合之前。
256 通道输入通过单次 Haar DWT 得到 LL/LH/HL/HH，保留原 REM 的四条卷积分支：

- LL：1x1 Conv/BN/SiLU。
- LH：1x1、1x3、3x1、3x3，各层 Conv/BN/SiLU。
- HL：1x1、3x1、1x3、3x3，各层 Conv/BN/SiLU。
- HH：1x1、3x3，各层 Conv/BN/SiLU。

令这些分支的输出为 F_LL、F_LH、F_HL、F_HH。原版为：

`Y = X + alpha * IDWT(F_LL, F_LH, F_HL, F_HH)`。

新增 per_band 模式为：

`Y = X + IDWT(alpha_LL*F_LL, alpha_LH*F_LH, alpha_HL*F_HL, alpha_HH*F_HH)`。

四个 alpha 都是零初始化、无约束、跨通道和空间共享的可学习标量，固定顺序为
LL/LH/HL/HH。使用原优化器学习率，无新增手动系数。不同子带可以学到不同幅值和符号；
没有预设低频必然抑制或高频必然增强。它是频带残差强度的解耦，不包含空间门控。

仍只执行一次 DWT 和一次 IDWT，奇数尺寸仍采用原 replicate padding 和裁剪。
没有额外的总 alpha。四个系数相同时，在线性 Haar 重建下与原共享 alpha 数学等价；
有限精度下仅允许运算顺序造成的舍入差异。初始四个系数为零，输出精确等于 X。
第一步更新四个 alpha；系数非零后，卷积分支开始获得非零梯度。

| 项目 | 原 REM | REM-per-band |
|---|---:|---:|
| 残差标量数 | 1 | 4 |
| 适配器可训练参数 | 2,823,681 | 2,823,684 |
| 全网络可训练参数 | 22,301,250 | 22,301,253 |
| DWT / IDWT 次数 | 1 / 1 | 1 / 1 |
| 四分支卷积结构与参数 | 原版 | 完全相同 |

原版默认 `rem_residual_mode: shared`，旧配置和共享 alpha 检查点保持原结构。
新版指定 `HybridEncoder.rem_residual_mode: per_band`；alpha 参数形状为 [4]。
恢复同一实验时使用保存配置，不将共享 alpha checkpoint 通过 `-r` 恢复为 per_band。

## 配置与冻结协议

新增配置均在 configs/deim_dfine/：

- deim_hgnetv2_m_sar_aircraft_102e_b28_old_server.yml：原冻结 102 轮 baseline 的服务器路径适配。
- deim_hgnetv2_m_sar_aircraft_rem_102e_b28_old_server.yml：原共享 alpha REM 的 102 轮配置。
- deim_hgnetv2_m_sar_aircraft_rem_per_band_58e_b28_old_server.yml：独立四系数、58 轮。
- deim_hgnetv2_m_sar_aircraft_rem_per_band_102e_b28_old_server.yml：独立四系数、102 轮。

DEIM-D-FINE-M / HGNetv2-B2、7 类 ID 0～6，train 3047 / val 442 / test 879。
seed 0，全局 batch 28，双卡各 14；输入基础及验证 640，训练多尺度 480～800。
AdamW 主 LR 3.5e-4、骨干非 BN LR 3.5e-5、WD 1e-4，EMA、SyncBN、AMP FP16、
FP32 loss/retry。COCO 初始化为 deim_dfine_hgnetv2_m_coco_90e.pth。
Decoder 4 层、采样点 [3,6,3]、reg_max 32；原 MAL、L1、GIoU、FGL、DDF 及 top300 后处理。

| 调度 | 58 轮 | 102 轮 |
|---|---|---|
| warmup_iter | 432 | 2000 |
| flat_epoch | 29 | 49 |
| no_aug_epoch | 8 | 12 |
| policy.epoch | [4,29,50] | [4,49,90] |
| mixup_epochs | [4,29] | [4,49] |
| stop_epoch | 50 | 90 |
| lr_gamma | 0.5 | 0.5 |

RandomZoomOut 和 RandomIoUCrop 在两个协议中都全程关闭；其余增强沿用对应冻结协议。
102 轮并非简单接续 58 轮训练。数据沿用现有 /home/zxy/sar/datasets/SAR-AIRcraft-1.0，
不需要重新准备标注。

## 服务器 Git 更新与首个正式运行

先查看工作树；若有改动或分叉，保留并处理后再更新，不使用强制覆盖。

```bash
cd /home/zxy/sar/repos/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/data2/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_102e_b28_old_server.yml
```

启动器创建全新的时间戳/随机后缀目录，记录配置、Git revision、seed、batch、输入、LR、
轮数、增强、EMA/AMP、初始化和评估协议。通过 nohup 在 GPU 0,1 上运行
torchrun --standalone --nproc_per_node=2 train.py，stdout/stderr 保存 console.log，
stdin 来自 /dev/null；先保存 $! 到 launcher.pid，再 disown。
以打印的 STARTED、PID、OUT 和随后日志判断是否实际启动。
两卡不同时空闲时，现有启动器退出，不排队、不停止正在运行的任务。

第一组结束且两卡空闲后，启动 58 轮 per_band：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/data2/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_per_band_58e_b28_old_server.yml
```

58 轮 per_band 优于原 REM 后，再启动其 102 轮验证：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/data2/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_per_band_102e_b28_old_server.yml
```

## 日志、GPU、进程与指标

重连后显式设置 OUT 为启动器打印的对应实验目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为实际实验目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C 退出日志跟踪后，读取最佳验证 AP 所在同一轮的完整指标：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
import yaml

run = Path(sys.argv[1])
cfg = yaml.safe_load((run / 'resolved_config.yml').read_text())
rows = [json.loads(s) for s in (run / 'log.txt').read_text().splitlines() if s.strip()]
by_epoch = {r['epoch']: r for r in rows if 'test_coco_eval_bbox' in r}
rows = [by_epoch[e] for e in sorted(by_epoch)]
best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
print('epochs=', cfg['epoches'], 'residual_mode=', cfg['HybridEncoder'].get('rem_residual_mode', 'shared'))
for label, row in [('best_AP_checkpoint', best), ('last_logged', rows[-1])]:
    print(label, 'epoch=', row['epoch'], 'AP/AP50/AP75/AP_S/AP_M/AP_L=',
          [round(v * 100, 4) for v in row['test_coco_eval_bbox'][:6]],
          'training_alpha_epoch_means=', {k:v for k,v in row.items() if k.startswith('train_rem_alpha')})
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

JSONL 为 log.txt；TensorBoard 为 summary/；训练 COCO 评估数据为 eval/latest.pth。
原 REM 日志仍是 train_rem_alpha；per_band 为 train_rem_alpha_ll/lh/hl/hh，
TensorBoard 对应 Module/rem_alpha_ll/lh/hl/hh。这些是训练参数的 epoch 均值，
不是验证时 EMA checkpoint 的系数。四系数数值本身也不能代替性能消融结论。
AP/AP50/AP75 为 COCO bbox 数组的 0/1/2 项，报告为百分数时乘 100。

## 恢复训练

原训练进程停止后，使用保存的 resolved_config.yml 和 -r last.pth：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为待恢复实验目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复日志为启动器打印的 resume_*.console.log，保留原 OUT；不同时使用 -t。
last.pth 仅在 stop_epoch 前更新：58 轮正常最后为 epoch49，102 轮为 epoch89，
均不是最终轮权重。收尾阶段中断时，现有恢复方式会从该保存点重新走后续阶段。

## 按验证 AP 选择 checkpoint 并评估测试集

所有指标取最佳验证 AP 对应的同一 checkpoint，不拼接各指标独立峰值。
使用对应 run 的保存配置，以相同评估口径比较：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为实际实验目录
CKPT=$(python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
import yaml

run = Path(sys.argv[1])
cfg = yaml.safe_load((run / 'resolved_config.yml').read_text())
rows = [json.loads(s) for s in (run / 'log.txt').read_text().splitlines() if s.strip()]
by_epoch = {r['epoch']: r for r in rows if 'test_coco_eval_bbox' in r}
best = max(by_epoch.values(), key=lambda r: r['test_coco_eval_bbox'][0])
stop = cfg['train_dataloader']['collate_fn']['stop_epoch']
print(run / ('best_stg2.pth' if best['epoch'] >= stop else 'best_stg1.pth'))
print(f"val epoch={best['epoch']} AP={best['test_coco_eval_bbox'][0] * 100:.4f}", file=sys.stderr)
PY
)
EVAL=$(mktemp -d "$OUT/test_eval_XXXXXX")
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$CKPT" --test-only --seed 0 --use-amp \
  --output-dir "$EVAL" --summary-dir "$EVAL/summary" \
  -u val_dataloader.dataset.ann_file=/home/zxy/sar/datasets/SAR-AIRcraft-1.0/COCO/test.json \
  > "$EVAL/console.log" 2>&1 < /dev/null &
EVAL_PID=$!
printf '%s\n' "$EVAL_PID" > "$EVAL/launcher.pid"
disown "$EVAL_PID"
printf 'EVAL=%s PID=%s CHECKPOINT=%s\n' "$EVAL" "$EVAL_PID" "$CKPT"
tail -f "$EVAL/console.log"
```

独立评估结果保存为 $EVAL/eval.pth，指标见 $EVAL/console.log。
结果表同时记录训练协议、对应 baseline 差值以及相对同协议原 REM 的差值。

## 本地功能验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_rem_per_band.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_radar_enhancement.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_rem_mechanism_ablations.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_pgd_rem.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sar_58e_protocol.py -v
```

针对本次改动验证共同初始化和 RNG、旧共享 alpha checkpoint、相等系数的输出/梯度等价、
四子带路由与单次 DWT/IDWT、初始恒等与两步学习、CPU BF16 autocast、stride-8 插入、
真实四层 Decoder 的普通/aux/DN 原损失训练、AdamW/EMA checkpoint 恢复、四系数日志，
以及 58/102 轮配置的协议继承和真实模型参数量。
本地验证不代表真实 SAR 数据训练结果，亦不替代服务器 CUDA FP16 性能测量。

2026-11-01 本地结果：新增 10 项、原 REM 5 项、机制对照 10 项、PGD-REM 6 项、
58 轮协议 5 项，共 36 项全部通过。真实配置工厂得到的参数量与上表一致。
启动器、本文 7 段 Bash 命令及 5 段嵌入 Python 的语法验证通过。
