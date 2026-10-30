# SAR-AIRcraft SDCP 实现与实验交接

## 实现边界和默认值

本模块实现框内尺度归一化证据融合、余弦原型分类、正样本概率边际损失，以及容噪非对称边界监督。
背景捷径、幅值捷径和标注差异对性能的具体贡献仍是待验证的机制假设；本实现不预先承诺 AP 收益。

分类器复用 CLCR 的 stride-8 特征采样器：中心坐标不变，在原预测框的 0.75 倍宽高内
采样 3×3 网格，平均池化为 F_in。启用 CLCR 时，同一次 grid_sample 的内部与外环证据
分别供 SDCP 和 CLCR 使用。训练的 pre-output 单独按其预测框采样；普通层与 DN 层均使用
各自当前预测框。采样点位于框内，但特征的感受野仍可包含背景；轴对齐网格也不等于语义部件对齐。

分类公式为 `q_fused = q + Proj([F_in, phi(log(w/h))])`，
`z = tau * normalize(q_fused) @ normalize(weight).T`。
默认 phi 是 1→16→16 的 ReLU MLP，Proj 是 272→256 的线性层，固定 tau=10，无类别偏置。
长宽比使用实际输入的像素宽高，由归一化框乘以输入宽高恢复，兼容非正方形输入。
融合及 L2 归一化以 FP32 计算，再返回原 Query dtype。余弦归一化消除融合向量整体模长的影响，
不保证任意影像辐射变化后特征方向完全不变；Query 仍有上下文信息。

每个 decoder 层独立使用余弦分类头；encoder 的粗粒度候选选择仍沿用原分类层。
输出顺序是原始余弦 logits → 原 LQE 标量修正 → 可选 CLCR 类别修正。
保留 `dec_score_head.N.weight` 的参数名及形状，删除原线性类别偏置，新增参数在 baseline
初始化完成后创建并隔离随机状态。开启 SDCP 会主动改变分类输出，不是恒等初始化。
COCO 初始化继续使用原 `-t` 的同名同形参数加载规则；80 类与 7 类的分类权重不直接等同。
默认每层余弦分类头 71,984 个参数；相对原线性头，四层合计增加 280,740 个可训练参数。
采样器无可训练参数，边际损失与容噪边界损失只在训练计算。

`loss_margin` 使用最终层校准前的余弦 logits，只作用于最终 Hungarian 匹配的普通正样本 Query：
`mean(relu(m - (sigmoid(z_y) - max_other(sigmoid(z)))) ** 2)`。
默认 m=0.2、损失权重 0.1；不额外叠加 aux/DN 边际项。双卡按全局正样本数归一化，
含某卡没有正样本及全局空正样本的情形。原 VFL/MAL 分类监督继续作用于原有输出路径。

框监督采用 xyxy 四条边的归一化误差 e。容差按每次实际输入尺寸换算：
左右边 epsilon=1/W，上下边 epsilon=1/H，因此 640 和 800 输入下均对应 1 像素。
定义 `u = relu(abs(e) - epsilon)`，边界惩罚为
`rho = 0.5*u**2/epsilon`（u<epsilon），否则为 `u-0.5*epsilon`。
容差内梯度为零，随后一个像素范围内平滑增大。若预测宽度大于真值宽度，左右两条边乘
`1 + lambda_over = 1.2`，上下边保持系数 1。该规则依据相对真值的超宽判断，不直接减去固定 2.75%。

SDCP 配置以权重 5 的 `loss_asym_box` **替代原权重 5 的 L1 `loss_bbox`**，
避免原严格 L1 抵消容噪效果。同时使用 detached 权重
`t_edge = clamp(u/epsilon, 0, 1)`：FGL 每条边的原 IoU 权重乘 t_edge 及上述非对称系数，
GIoU 权重乘四边 t_edge 的均值。GIoU 仍是耦合四边的整体损失；部分边超过容差时，
不能将它解释为其他边严格零梯度。所有边都在容差内时，三项 GT 边界监督均为零。
FDR 的目标分桶不改变，DDF 教师分布蒸馏保持原规则。上述边界处理覆盖最终、aux、pre、encoder 和 DN 的原监督路径。
分类/采样梯度仍可传入预测框，因此“容差内零梯度”仅指这些 GT 边界监督，不指模型总梯度。

参数均可配置：`DFINETransformer.use_sdcp/sdcp_aspect_dim/sdcp_temperature`，
`DEIMCriterion.use_sdcp/sdcp_margin/sdcp_epsilon_px/sdcp_lambda_over` 及两个新增 loss 权重。
已有配置默认关闭 SDCP，不修改已报告 baseline 的实验设定。

## 文件与实验协议

- `engine/deim/dfine_decoder.py`：共享采样器、余弦分类头、逐层路由、原始 logits 输出和部署裁剪兼容。
- `engine/deim/deim.py`：传递实际输入尺寸。
- `engine/deim/deim_criterion.py`：全局正样本边际均值、容噪非对称边界损失及 FGL/GIoU 权重。
- `tools/run_sar_module_experiment.sh`：归档 SDCP 参数、损失权重与完整解析配置。
- 单模块：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_sdcp_58e_b28_old_server.yml`。
- 三模块：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pgd_rem_clcr_sdcp_58e_b28_old_server.yml`。
- `tests/test_sdcp.py`、`tests/check_sdcp_ddp.py`：公式、真实损失路由、模块组合、部署及分布式梯度验证。

两配置继承同一 58 轮 baseline；三模块配置额外启用 PGD-REM 和 CLCR。
协议：DEIM-D-FINE-M / HGNetv2-B2、7 类 ID 0–6、train3047 / val442 / test879，
seed0、GPU0/1、全局 batch28（每卡14）、58轮、基础/验证640、训练多尺度480–800；
warmup432、flat29、no_aug8、lr_gamma0.5，增强 policy [4,29,50]、mixup [4,29]、stop50。
AdamW 主 LR3.5e-4、骨干非 BN LR3.5e-5、WD1e-4；Mosaic/MixUp、EMA、AMP FP16、
FP32 loss/retry、query×class 展平 top300 后处理继承原协议。
训练均从同一 COCO M 权重 `-t` 开始，数据沿用现有 SAR-AIRcraft 目录及 0–6 标注，无需重新转换。
启动器记录 Git revision、seed、全局 batch、输入尺寸、优化器、增强、EMA/AMP、初始化及评估配置。

## 服务器更新和启动

若服务器工作树有改动或分叉，先检查并保留改动；不强制覆盖。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_sdcp_58e_b28_old_server.yml
```

启动器创建新目录，用 `CUDA_VISIBLE_DEVICES=0,1`、`torchrun --standalone --nproc_per_node=2 train.py`；
以 `nohup bash "$COMMAND" > "$LOG" 2>&1 < /dev/null &` 后台运行，保存 `$!` 至 launcher.pid 后 disown。
首次日志为 console.log。前一实验结束、两卡可用后，三模块另起新目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pgd_rem_clcr_sdcp_58e_b28_old_server.yml
```

## 日志、指标、TensorBoard 和恢复

重连后显式设置 OUT 为启动器打印的实际目录，并激活环境：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际运行目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C 退出日志跟踪，读取最佳 AP、最近损失及 alpha/beta：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(line) for line in (Path(sys.argv[1]) / 'log.txt').read_text().splitlines() if line.startswith('{')]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
for label, row in [('best_val', best), ('last_logged', rows[-1])]:
    details = {k:v for k,v in row.items() if k in ('train_loss_margin', 'train_loss_asym_box', 'train_pgd_rem_alpha') or k.startswith('train_clcr_beta_')}
    print(label, 'epoch=', row['epoch'],
          'AP/AP50/AP75=', [round(v * 100, 4) for v in row['test_coco_eval_bbox'][:3]],
          'losses_and_scales=', details)
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

JSONL 指标在 log.txt；COCO bbox 第0/1/2项是 AP/AP50/AP75，百分数报告乘100。
TensorBoard 在 summary/，训练 COCO 数据在 eval/latest.pth。

原进程停止后恢复同一实验：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原运行目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复读取保存的 resolved_config.yml，使用 `-r last.pth`，不同时传 `-t`。
恢复日志为启动器打印的 resume_*.console.log。当前协议 last.pth 正常最后更新于 epoch49，
不是最后一轮权重；恢复到该保存点后继续训练。

## 按验证 AP 选择权重并评估测试集

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原运行目录
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

按实际验证 AP 选择 best_stg1.pth 或 best_stg2.pth，不能直接把第二阶段文件当作最优。
独立评估数据在 `$EVAL/eval.pth`，指标见控制台日志。

## 本地功能验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sdcp.py -v
conda run --no-capture-output -n pytorch python tests/check_sdcp_ddp.py
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_clcr.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_query_prototypes.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_pgd_rem.py -v
```

验证关注公式与梯度、640/800 及非方形输入的像素容差、共享采样、最终/aux/pre/encoder/DN
损失路由、三模块联合优化、checkpoint 恢复、部署和配置协议。两 CPU rank 的 Gloo 验证
比较实际 DDP 梯度与串行全局正样本均值，覆盖 (1,0)、(2,1)、(0,0) 三种正样本分布。
结果：SDCP 14 项、CLCR 10 项、原 decoder/MPQ 回归 7 项、PGD-REM 6 项，合计 37 项全部通过；
CPU 双进程三种正样本分布的梯度比较全部通过。启动器、本文 6 段 Bash 命令及嵌入 Python
片段语法检查通过，未执行服务器命令。
本地验证不等于真实数据训练；CUDA 双卡显存、吞吐与检测收益需服务器实验确认。
