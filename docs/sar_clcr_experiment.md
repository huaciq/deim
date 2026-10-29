# SAR-AIRcraft：CLCR 可靠性校准头

CLCR按用户给定公式扩展现有LQE，将类别竞争、局部内外对比和原几何统计联合用于
类别特定的logit残差。它通过现有检测损失训练，不增加监督标注或额外损失。

## 确切计算与接入

每个decoder层沿用原LQE的20维几何特征：四条边各取分布概率top4及其均值。
原 `reg_conf` 参数名、维度和输出标量保持原样，可接入原COCO初始化权重。

对于当前层尚未加LQE的分类logits `z[B,Q,7]`：

```text
p = sigmoid(z)
competition[c] = [p[c], p[c]-max(p[other]), p[c]/(sum(p)+1e-6), H]
H = -sum(p_normalized * log(p_normalized)) / log(7)

F_in   = mean(grid_sample(P3, 0.75*box内的3×3网格))
F_ring = mean(grid_sample(P3, 1.05*box四边各3个点))
local16 = MLP_512->64->16([F_in, F_in-F_ring])

delta[c] = MLP_40->64->1([competition[c], local16, edge20])
calibrated_logits[c] = z[c] + reg_conf(edge20) + beta * delta[c]
```

- `P3`是编码器/FPN输出、decoder投影后的stride-8特征，默认256通道。
  从现有flatten memory取视图重建一次；不复制成每个Query一张特征图。
- 当前层的FDR预测框 `inter_ref_bbox` 提供归一化cxcywh坐标。缩放只作用于宽高，
  中心保持不变。内部9点的相对偏移为 `{-0.375,0,0.375}²`。
  外环上下边为 `y=±0.525, x∈{-0.35,0,0.35}`，左右边交换x/y，
  共12个不同点，不重复采角点。全部21点在每层的一次grid_sample中完成。
- 采用双线性采样、`align_corners=False` 和 `padding_mode=border`；越界点使用边界延拓。
  网格与采样在FP32中计算；竞争统计也使用FP32，熵由log-sigmoid/log-softmax计算，
  避免极负logits下归一化概率下溢。MLP沿用训练AMP设置，残差返回原logit dtype。
- 两个MLP均为两层，隐藏宽度64，ReLU；局部输出固定16维。
  校准MLP在七个类别之间共享权重，每个类别输入自己的四维竞争统计，输出七个不同修正量。
  它不是把同一标量再次广播给七个类别，也没有硬编码“负margin必须降分”的规则。
- 每个LQE层有独立CLCR和一个零初始化标量beta。其余新增权重正常初始化，
  初始输出与原LQE严格一致；beta先学习，随后局部和校准MLP接收有效梯度。
- 不detach原logits、几何统计、当前预测框或局部特征，保持给定公式的端到端梯度。
  模块不直接修改框坐标；训练时分类损失可以通过采样坐标影响框分支。
- 训练时所有decoder层的普通Query和DN Query均使用各自当前预测框校准，
  沿用原最终、aux、DN分类损失。encoder预选和pre-output不经过LQE，因此不增加CLCR。
  原定位蒸馏继续使用最终层logits作为teacher质量权重，现为校准后的logits。
- 推理时仅计算选定 `eval_idx` 层的LQE/CLCR；部署裁剪保留该层校准。
  新模块在完整baseline初始化之后创建，并隔离随机数状态，保留共同权重和后续DN随机序列。

默认256通道、20维几何统计、隐藏宽度64下，每层CLCR新增36,562个可训练参数；
DEIM-M四层合计146,248个。普通推理只执行输出层的CLCR，部署裁剪后也只保留该层。

“内外对比可以识别杂波/框偏宽”是待验证的机制假设。采样的是学习到的特征，
不是原始SAR复数散射信号；此前离线重排收益不能直接视作本模块的训练收益。

## 文件、配置和固定协议

- `engine/deim/dfine_decoder.py`：新增CLCR，扩展LQE，接入逐层当前框和stride-8特征。
- `engine/solver/det_engine.py`：记录每层 `train_clcr_beta_l0` 到 `train_clcr_beta_l3`。
- `tools/run_sar_module_experiment.sh`：记录CLCR开关、隐藏宽度及完整解析配置。
- `tests/test_clcr.py`：竞争公式、采样几何、梯度、aux/DN、组合训练和部署验证。
- 单模块配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_clcr_58e_b28_old_server.yml`。
- 组合配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pgd_rem_clcr_58e_b28_old_server.yml`。

两份配置继承同一58轮baseline；组合配置仅额外开启模块一PGD-REM。
已有REM、PACFG、PGD-REM和baseline配置保持原行为。

协议：DEIM-D-FINE-M / HGNetv2-B2、7类ID0–6、train3047 / val442 / test879、
seed0、GPU0/1、全局batch28（每卡14）、58轮、基础/验证640、训练多尺度480–800；
warmup432、flat29、no_aug8、lr_gamma0.5；增强policy [4,29,50]、mixup [4,29]、stop50。
AdamW主LR3.5e-4、骨干非BN LR3.5e-5、WD1e-4；Mosaic/MixUp、EMA、AMP FP16、
FP32 loss/retry、原损失和query×class展平top300后处理均继承。
两种实验均从相同COCO M权重以 `-t` 新训，不从单模块SAR权重续训。
数据沿用现有目录及0–6标注，无需重新转换。启动器归档配置、Git revision、seed、
优化器/增强/EMA/AMP、初始化路径和命令；与同58轮baseline比较。

## 服务器更新及单模块启动

先检查Git状态；若有本地改动或分叉，保留并检查，不强制覆盖。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_clcr_58e_b28_old_server.yml
```

启动器新建实验目录，使用 `CUDA_VISIBLE_DEVICES=0,1`、
`torchrun --standalone --nproc_per_node=2 train.py`，实际后台命令为
`nohup bash "$COMMAND" > "$LOG" 2>&1 < /dev/null &`，保存 `$!` 至 `launcher.pid` 后执行 `disown`。
不覆盖已有运行，首次日志为 `console.log`。

前一实验结束、两卡可用后，组合实验另起新目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pgd_rem_clcr_58e_b28_old_server.yml
```

## 监控、指标及恢复

重连后显式激活环境，将OUT改为启动器打印的实际目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际CLCR或组合运行目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C退出日志跟踪后，查看最佳指标与残差系数：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(line) for line in (Path(sys.argv[1]) / 'log.txt').read_text().splitlines() if line.startswith('{')]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
for label, row in [('best_val', best), ('last_logged', rows[-1])]:
    scales = {k:v for k,v in row.items() if k.startswith('train_clcr_beta_') or k == 'train_pgd_rem_alpha'}
    print(label, 'epoch=', row['epoch'],
          'AP/AP50/AP75=', [round(v * 100, 4) for v in row['test_coco_eval_bbox'][:3]],
          'scales=', scales)
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

JSONL指标在 `log.txt`；`test_coco_eval_bbox`第0/1/2项为AP/AP50/AP75，百分数报告乘100。
TensorBoard事件在 `summary/`，COCO数据在 `eval/latest.pth`。

原进程停止后恢复同一实验：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原CLCR或组合运行目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复使用保存的 `resolved_config.yml` 和 `-r last.pth`，不同时传 `-t`；
日志为启动器打印的 `resume_*.console.log`。本协议 `last.pth` 正常最后更新于epoch49，
并非最终epoch权重。

## 选取最佳验证权重并评估测试集

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原CLCR或组合运行目录
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

按实际验证AP选阶段权重，不单凭阶段名判断。独立评估数据在 `$EVAL/eval.pth`，指标见控制台日志。

## 本地功能验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_clcr.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_query_prototypes.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_pgd_rem.py -v
```

覆盖竞争统计解析值及并列概率、极值熵、LQE原项及类别修正、采样几何和批次隔离、
CPU/BF16下零beta启动及所有证据梯度、初始化/aux/DN一致性、逐层当前框与stride-8路由、
现有检测损失对CLCR和PGD-REM的梯度、checkpoint恢复、部署裁剪及两配置协议一致性。
结果：CLCR 10项、原decoder/MPQ相关回归7项、PGD-REM回归6项，合计23项全部通过。
启动器、本文6段Bash命令和其中的Python片段语法检查通过，没有执行服务器命令。
本地不加载真实数据或COCO权重，未进行CUDA双卡训练；显存、速度及检测增益需服务器实验确认。
