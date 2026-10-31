# SAR-AIRcraft REM + CLCR-Lite：输入detach与epoch30延迟启用

## 方法B的作用与边界

本次执行用户提出的端到端方法B：CLCR-Lite全部8维输入detach，epoch>=30才启用残差。
这是降低早期梯度耦合的实验假设，不保证AP不下降，也不等价于冻结整个检测器。

原匹配器HungarianMatcher.forward已经使用torch.no_grad()，匹配过程本身不参与反向传播。
输入detach只删除校准增量对输入的导数；启用后，校准后的数值仍传入匹配器，可能改变分配。
对 z_hat=z+LQE+delta(stopgrad(stats))，原始z/LQE仍接收分类损失梯度，
而该梯度的数值会随delta改变；DDF教师权重等沿用原来读取校准logits的路径。
因此本版本不宣称匹配固定、主网梯度完全隔离或绝对不掉点。

参考PyTorch官方说明：
[detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)、
[grad=None的优化器语义](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)、
[DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。
实际行为另由本地PyTorch功能验证确认。

## 实现

- engine/deim/dfine_decoder.py：CLCRLite新增detach_inputs/start_epoch，DFINETransformer透传配置。
- engine/solver/det_engine.py：每轮同步主模型和EMA启用状态，在优化器更新前清理未启用头的梯度。
- tools/run_sar_module_experiment.sh：归档两个新开关和关闭期更新规则。
- tests/test_clcr_lite_detach30.py、tests/check_clcr_lite_detach30_ddp.py：梯度、冻结、匹配、EMA及双进程验证。

共享头仍是Linear(8,16)->SiLU->Linear(16,1)，仅输出层零初始化，beta=1，共162参数。
8维仍为[p,margin,H,边缘集中度均值,边缘集中度标准差,lnw,lnh,ln(w/h)]；
在输入MLP前对整个8维descriptor执行detach，不改统计量定义。
原LQE的完整20维统计和几何修正保留正常梯度，原分类/框回归损失正常训练。
普通Query、aux和DN的四层校准统一使用同一个头和同一启用状态，无新增辅助loss。

epoch采用日志的0-based编号：0..29关闭，30起启用，即完成前30轮后、从第31个训练epoch开启。
关闭期不计算MLP输入和修正，只返回严格零残差；用所有头参数连接的零值保留DDP梯度hook。
在backward完成后、AMP unscale/梯度裁剪/optimizer.step之前，把这些零梯度置None，
从而跳过AdamW权重衰减和动量更新。参数始终requires_grad=True，不在DDP包装后临时增减可训练参数。
因此首层随机初始化、输出层零值和beta=1在关闭期保持不变，epoch30开始正常优化。

延迟配置额外保存一个bool启用buffer，参数量仍为162；加载时同步Python启用标记，
独立评估/部署无需手工提供epoch，早期checkpoint保持关闭、后期checkpoint保持开启。
EMA的bool不做数值平均：每轮显式同步主模型和EMA；模型/EMA各自保存正确状态。
无增强阶段重载best_stg1后，下一轮入口按当前epoch再次设置开关。
EMA衰减仍沿用原协议，不重启全网络EMA；校准头有效训练窗口及EMA适应时间更短，收益需实测。

原CLCR-Lite默认detach_inputs=False/start_epoch=0，原参数和state_dict键保持不变。
本次新增两配置，均位于configs/deim_dfine/：

| 配置名主体 | REM | 输入detach | 校准开始epoch |
|---|---|---|---:|
| rem_clcr_lite_58e_b28_old_server | 原REM | 否 | 0 |
| rem_clcr_lite_detach30_58e_b28_old_server | 原REM | 是 | 30 |

第二组相对第一组同时改变两个因素，只能先判断整体方案效果；分离归因时再补detach-only/delay-only。
它们与已有单REM结果一起比较；原Lite单模块结果不能代替REM+Lite即开组合对照。

## 固定协议

DEIM-D-FINE-M / HGNetv2-B2、7类ID0–6，train3047/val442/test879；seed0、58轮、
全局batch28（双卡每卡14）、基础/验证640、训练多尺度480–800；
warmup432、flat29、no_aug8、lr_gamma0.5，增强policy [4,29,50]、mixup [4,29]、stop50；
AdamW主LR3.5e-4、骨干非BN LR3.5e-5、WD1e-4，EMA、SyncBN、AMP FP16及FP32 loss/retry。
RandomZoomOut和RandomIoUCrop仍全程关闭；REM为已报告的haar_directional，W-FDR未启用。
从同一COCO M权重-t新训并使用新优化器，数据及0–6标注沿用现有目录，无需重新准备。
启动器记录解析配置、Git revision、seed、batch、输入、LR、增强、EMA/AMP、初始化和评估协议。
每组按最佳val AP选一个checkpoint，再同时报告其AP/AP50/AP75/AP_S/AP_M/AP_L。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_clcr_lite_detach30_58e_b28_old_server.yml
```

启动器创建时间戳及随机后缀的新目录，使用CUDA_VISIBLE_DEVICES=0,1及
torchrun --standalone --nproc_per_node=2 train.py，以 -t 做COCO初始化。
正式进程由nohup后台执行，stdout/stderr写console.log、stdin来自/dev/null，
先保存$!至launcher.pid，再disown。启动器打印STARTED OUT=... PID=... LOG=...。
原启动器会检查两卡是否空闲；不空闲时不启动、不排队、不停止已有进程。

第一组结束后，如需与原端到端组合比较，使用独立配置启动对照：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/data2/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_clcr_lite_58e_b28_old_server.yml
```

## 查看日志、进程及指标

重连后激活环境，并把OUT替换为启动器打印的实际延迟校准实验目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为实际REM_CLCR_LITE_detach30目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C退出日志跟踪后读取最佳AP检查点的全部指标，以及REM残差、校准beta及启用标记：

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
    print({k: v for k, v in row.items() if k in ('train_rem_alpha', 'train_clcr_lite_beta', 'train_clcr_lite_active')})
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
export OUT=/data2/zxy/sar/experiments/替换为原REM_CLCR_LITE_detach30目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复日志为启动器打印的resume_*.console.log；保存恢复PID和命令。
当前solver的last.pth只在无增强阶段前更新，本协议正常最后更新于epoch49，不是第57轮权重。

## 按验证AP选权重并评估测试集

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为原REM_CLCR_LITE_detach30目录
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

## 本地功能验证与限制

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_clcr_lite_detach30.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_clcr_lite.py -v
conda run --no-capture-output -n pytorch python tests/check_clcr_lite_detach30_ddp.py
```

验证8维梯度截断、直接logit梯度和匹配仍可变化、29/30边界、AdamW真冻结、真实FP32/CPU BF16
训练循环及EMA同步、REM关闭期输出/损失/梯度等价、checkpoint/部署、旧Lite兼容及协议。
DDP验证使用两进程CPU Gloo及四层共享头，检查关闭到启用切换与串行全局均值梯度一致。
本地未执行真实数据58轮训练或CUDA双卡性能测量，服务器启动以STARTED、PID和日志为准。

验证结果：新增9项、原Lite回归11项、58轮协议5项，共25项全部通过。
两进程Gloo在epoch28/29/30/31通过：关闭期参数和AdamW状态不变，启用后梯度与串行全局均值一致。
真实训练循环的FP32/CPU BF16 AMP与EMA同步、独立匹配反例、保存/部署状态均通过验证。
两种REM+Lite配置的全网络可训练参数均22,301,412；启动器、本文6段Bash及5段嵌入Python语法检查通过。
