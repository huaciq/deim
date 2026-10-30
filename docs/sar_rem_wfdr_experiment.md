# SAR-AIRcraft REM + W-FDR：58轮组合实验

## 组合与比较口径

配置：configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_wfdr_58e_b28_old_server.yml。
单父级继承已报告的W-FDR单模块配置，仅新增HybridEncoder.use_rem及独立输出/文档路径。
不改任何模块实现；REM和W-FDR的结构、插入位置、损失权重均沿用单模块实验。

- REM：原RadarEnhancementModule，stride-8投影后、FPN/PAN融合前，256通道，
  单次Haar分解/重建与四个原卷积分支，零初始化残差alpha。
- W-FDR：原loss_local中的非均匀桶W1，保留原FGL和DDF，
  wfdr_weight=0.5、weight_dict.loss_wfdr=0.15，合计为0.15*(DFL+0.5*W1)。
  作用于普通最终层、aux及DN的FDR分布，不增加推理参数和计算。

REM+W-FDR的检测网络与REM单模块一致。PACFG、PGD-REM、CLCR/Lite、SDCP、SPI及OST-DA沿用关闭状态。
比较组为baseline / REM / W-FDR / REM+W-FDR；既有前三组结果保留。
组合从同一COCO M权重 -t 新训、使用新优化器，不接着REM或W-FDR训练权重继续训。
按验证集最佳AP选择一个checkpoint，并同时报告其AP/AP50/AP75/AP_S/AP_M/AP_L，
不把不同轮次的独立峰值拼成一行；组合首先与最好的单模块REM比较。

## 固定协议

DEIM-D-FINE-M / HGNetv2-B2、7类ID0–6，train3047 / val442 / test879；
seed0、58轮、全局batch28（双卡每卡14）、基础及验证640、训练多尺度480–800；
warmup432、flat29、no_aug8、lr_gamma0.5；增强policy [4,29,50]、mixup [4,29]、stop50；
AdamW主LR3.5e-4、骨干非BN LR3.5e-5、WD1e-4、EMA、SyncBN、AMP FP16、FP32 loss/retry。
RandomZoomOut和RandomIoUCrop仍全程关闭；其他增强沿用既有58轮协议。
标注及图像沿用现有SAR-AIRcraft-1.0路径，无需重新准备数据。
启动器归档解析配置、Git revision、seed、batch、输入、LR、增强、EMA/AMP、初始化和评估协议。

## Git更新与双卡新训

若服务器工作树有改动或分叉，先检查并保留，不强制覆盖。执行：

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_wfdr_58e_b28_old_server.yml
```

启动器创建时间戳及随机后缀的新目录，使用CUDA_VISIBLE_DEVICES=0,1及
torchrun --standalone --nproc_per_node=2 train.py，以 -t 做COCO初始化。
正式进程由nohup后台执行，stdout/stderr写console.log、stdin来自/dev/null，
先保存$!至launcher.pid，再disown。启动器打印STARTED OUT=... PID=... LOG=...。
原启动器会检查两卡是否空闲；不空闲时不启动、不排队、不停止已有进程。

## 查看日志、进程及指标

重连后激活环境，并把OUT替换为启动器打印的实际组合实验目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为实际REM_WFDR目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C退出日志跟踪后读取最佳AP检查点的全部指标，以及REM残差和W-FDR损失：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

by_epoch = {}
for line in (Path(sys.argv[1]) / 'log.txt').read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if 'test_coco_eval_bbox' in row:
        by_epoch[row['epoch']] = row
rows = [by_epoch[e] for e in sorted(by_epoch)]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
for label, row in [('best_AP_checkpoint', best), ('last_logged', rows[-1])]:
    print(label, 'epoch=', row['epoch'], 'AP/AP50/AP75/AP_S/AP_M/AP_L=',
          [round(value * 100, 4) for value in row['test_coco_eval_bbox'][:6]])
    print({k: v for k, v in row.items() if k == 'train_rem_alpha' or k.startswith('train_loss_wfdr')})
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

每轮JSONL指标在log.txt；COCO bbox第0/1/2项为AP/AP50/AP75，0..1值报告百分数时乘100。
TensorBoard目录summary/，训练评估数据eval/latest.pth。

## 恢复原组合运行

仅在原进程停止后执行，使用保存的resolved_config.yml及 -r last.pth，不同时传 -t：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为原REM_WFDR目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复日志为启动器打印的resume_*.console.log；保存恢复PID和命令。
当前solver的last.pth只在无增强阶段前更新，本协议正常最后更新于epoch49，不是第57轮权重。

## 按验证AP选权重并评估测试集

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为原REM_WFDR目录
CKPT=$(python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
import yaml

run = Path(sys.argv[1])
cfg = yaml.safe_load((run / 'resolved_config.yml').read_text())
by_epoch = {r['epoch']: r for s in (run / 'log.txt').read_text().splitlines()
            if s.startswith('{') for r in [json.loads(s)] if 'test_coco_eval_bbox' in r}
best = max(by_epoch.values(), key=lambda row: row['test_coco_eval_bbox'][0])
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

按实际验证AP选best_stg1.pth/best_stg2.pth；独立评估数据在$EVAL/eval.pth，指标见控制台。

## 本地验证与限制

验证组合配置与W-FDR单模块仅差REM开关，且与REM单模块相比仅增加既定W-FDR损失；
通过真实配置工厂核对模型及损失，确认网络参数与REM单模块一致。
使用真实Encoder/Decoder/Criterion路径检验普通、aux和DN损失，以及W1对REM和FDR的梯度连接。
本地不包含真实数据58轮训练或CUDA双卡性能测试；服务器是否启动以STARTED行、PID及日志为准。

验证结果：组合网络与REM单模块的初始化权重逐项一致，可训练参数同为22,301,250。
FP32及CPU BF16 AMP路径均通过两次参数更新；4层Decoder的8项W-FDR损失（最终1、aux3、DN4）
均生效，第二步W1对REM残差alpha、方向卷积和FDR回归层的独立梯度均非零且有限。
启动器、本文5段Bash命令及5段嵌入Python语法检查通过，未执行服务器命令。
