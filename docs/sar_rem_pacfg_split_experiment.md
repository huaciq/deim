# SAR-AIRcraft：REM stride-8 + PACFG stride-16 位置消融

本次改变 PACFG 的插入位置：REM 仍处理 stride-8 投影特征，PACFG 改为独立处理
stride-16 投影特征，随后两路进入现有 FPN/PAN。两个模块的内部实现保持不变。

## 当前串联、分层与可选并联

已有串联在 `HybridEncoder.forward` 的 FPN 之前执行：

```text
stride-8 投影特征 -> REM -> PACFG ----+
stride-16 投影特征 -----------------+-> 原 FPN/PAN -> decoder
stride-32 编码特征 -----------------+
```

本次分层配置执行：

```text
stride-8 投影特征  -> REM -----------+
stride-16 投影特征 -> PACFG ---------+-> 原 FPN/PAN -> decoder
stride-32 编码特征 -----------------+
```

640输入时，两模块输入分别为 `[B,256,80,80]` 和 `[B,256,40,40]`。
REM输出不再作为PACFG输入。FPN仍先融合stride-32和增强后的stride-16，
再向增强后的stride-8传递信息；PAN和三个输出尺度保持原样。
假设是让两者在不同尺度上处理结构，并交给已有FPN融合；效果尚待实验验证。
若此方案有效，正式消融还需补PACFG@stride-16单模块，区分位置收益与组合收益。

并联可作为后续独立对照，目前只给设计说明，没有加入并联代码或配置：
同一份stride-8特征X分别进入REM和PACFG，融合残差增量。
现有模块输出为 `R(X)=X+alpha_r*r(X)`、`P(X)=X+alpha_p*p(X)`，建议最小实现为
`Y=X+(R(X)-X)+(P(X)-X)`。两路沿用内部可学习alpha，初始输出仍为X；
直接相加完整的R(X)和P(X)会把原始X重复计算。并联检验同尺度独立处理是否有益，
与本次跨尺度位置消融回答不同问题，不预设其性能更高。

## 实现与固定协议

- `HybridEncoder.pacfg_feat_index` 默认0，保留已有单模块和串联实验行为；新配置设为1。
- 模块权重名称、参数量、初始化和剩余网络结构不变；运行元数据新增该位置索引。
- 新配置继承旧串联配置，只覆盖位置、输出目录和本说明文档。
- 配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_s8_pacfg_s16_58e_b28_old_server.yml`。

改动文件：`engine/deim/hybrid_encoder.py` 增加位置选项；
`tools/run_sar_module_experiment.sh` 记录该选项；
`tests/test_rem_pacfg_combination.py` 增加分层路由验证；另新增上述配置和本交接文档。

协议：DEIM-D-FINE-M / HGNetv2-B2、七类0–6、train3047 / val442 / test879、seed0、
GPU0/1、全局batch28（每卡14）、基础/验证尺寸640、训练多尺度480–800、58轮、
warmup432、flat29、no_aug8、lr_gamma0.5；policy [4,29,50]、mixup [4,29]、stop50；
AdamW 主LR3.5e-4、骨干非BN LR3.5e-5、WD1e-4。
Mosaic/MixUp、EMA、AMP FP16、FP32 loss/retry、原损失和评估规则全部继承。
从同一COCO M权重用 `-t` 新建训练，不继续训练旧串联权重；数据无需重新准备。
启动器归档完整配置、Git revision、seed、超参数、初始化路径和命令，
同时记录 `train_rem_alpha` 与 `train_pacfg_alpha`。

## 服务器更新与启动

有未提交改动或分叉时先检查并保留现有文件，不强制覆盖。

```bash
cd /home/zxy/sar/repos/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_s8_pacfg_s16_58e_b28_old_server.yml
```

启动器创建全新目录，使用 `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2`，
通过 `nohup ... > console.log 2>&1 < /dev/null &` 启动；保存 `$!` 到 `launcher.pid` 后 `disown`。
当前开发环境无法直连服务器，因此尚未远程启动训练或验证CUDA双卡效果。

## 日志、指标、TensorBoard与恢复

重连后先激活环境，将OUT设为启动器打印的实际目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际REM_S8_PACFG_S16训练目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C退出日志跟踪后运行TensorBoard：

```bash
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

`log.txt` 为JSONL指标；`test_coco_eval_bbox`第0/1/2项为验证AP/AP50/AP75，
第3/4/5项为AP_S/AP_M/AP_L，报告百分数时乘100。
TensorBoard事件在 `summary/`，训练COCO评估数据在 `eval/latest.pth`。
仅在原训练进程已停止时恢复：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原REM_S8_PACFG_S16训练目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复使用保存的解析配置和 `-r last.pth`，不同时传 `-t`；
查看启动器打印的 `resume_*.console.log`。当前 `last.pth` 正常最后更新于epoch49，不是最终权重。

## 按验证AP选择权重与测试集评估

先在验证集比较插入方式，候选确定后再评估测试集。
以下按日志最佳val AP选择阶段权重，不按阶段名判断优劣：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原REM_S8_PACFG_S16训练目录

CKPT=$(python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
import yaml

run = Path(sys.argv[1])
cfg = yaml.safe_load((run / 'resolved_config.yml').read_text())
rows = [json.loads(line) for line in (run / 'log.txt').read_text().splitlines() if line.startswith('{')]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
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

独立测试数据在 `$EVAL/eval.pth`，指标在对应控制台日志。

## 本地验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_rem_pacfg_combination.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_phase_aligned_frequency.py -v
```

验证关注新旧初始化、默认串联行为、两路独立输入、增强特征进入对应FPN层、
CPU/CPU autocast梯度与输出尺度，以及新配置只改变PACFG位置。
结果：5项组合验证和5项PACFG回归全部通过；Python、启动器及5段Bash交接命令语法检查通过。
本地没有加载真实图片/COCO权重或启动训练；显存、耗时和检测收益需服务器实验确认。
