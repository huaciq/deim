# SAR-AIRcraft W-FDR 实现与实验交接

## 公式与仓库接入

实际监督入口是 `engine/deim/deim_criterion.py::DEIMCriterion.loss_local`，
现有双点交叉熵在 `unimodal_distribution_focal_loss`，名为 loss_fgl；教师分布蒸馏另记 loss_ddf。
新增 `engine/deim/wasserstein_fdr.py::WassersteinFDRLoss`，在原 FGL 上叠加 GT 分布的 W1 项，
不替换教师 KL 蒸馏。训练默认 reg_max=32、33个桶，不改变 FDR 分桶、解码或推理输出。

使用模型当前 up、reg_scale 调用原 `weighting_function`，得到完整的非均匀桶坐标 v。
`p=softmax(pred_corners)`，目标 q 在相邻桶上按物理坐标线性插值，
`W1=sum(abs(cumsum(p)[:-1]-cumsum(q)[:-1]) * (v[1:]-v[:-1]))`。
使用精确 L1-CDF 形式，不引入 Smooth-L1 近似、Sinkhorn 或高斯方差超参数。
网格是 D-FINE 的边偏移坐标单位；没有额外乘框宽高或像素尺寸，也不将桶编号距离当作实际桶跨度。

相较示例代码，需要适配两点：

1. 原 `bbox2distance/translate_gt` 返回的是左桶索引和独立的 `weight_right/weight_left`，
   不是把插值比例编码在索引的小数部分。集成路径直接复用这两个权重，保证 W1 与原 DFL 的 q 一致。
2. 保留最后区间 [reg_max−1, reg_max] 和最后桶 reg_max。示例中将目标截为
   reg_max−1−1e-6 会误删该区间。已有上界溢出标签 31.9 配合 right_weight=1 仍正确表示第32桶。

独立调用保留示例的四个位置参数；未提供插值权重时，target_corners 解释为 [0,reg_max]
中的连续桶索引。传入仓库 bbox2distance 的结果时，必须同时使用 `weight_left=` 与 `weight_right=`。
默认返回 `loss_weight * mean(iou_weight * W1)`；空匹配返回保留梯度连接的0。
集成路径设置 `avg_factor=num_boxes`，按原 FGL 的 `sum(weighted_edges)/num_boxes` 归一化，
包括原 union 匹配集、跨卡 box 归一化及 DN 分组，不额外除4或重复乘双卡数。

默认 `wfdr_weight=0.5` 即 lambda_w，保持原 FGL 外部权重0.15，并设置新项外部权重0.15：

```text
loss_fgl  = 0.15 * sum(iou_weight * DFL) / num_boxes
loss_wfdr = 0.15 * 0.5 * sum(iou_weight * W1) / num_boxes
```

因此两项之和对应 `0.15 * (DFL + 0.5 * W1)`，lambda_w只乘一次。
`use_wfdr` 默认 False；启用时必须显式配置 weight_dict.loss_wfdr，避免新损失被权重过滤静默丢弃。
lambda_w可配为0进行数值对照。新增损失无可训练参数、无持久缓冲区，不消费模型初始化随机数。
概率、CDF、网格及W1在FP32中计算；GT、插值权重、桶网格和IoU权重均作为监督量detach。

最终层、所有aux层及DN层只要具有 pred_corners 就应用W1；pre、encoder、DN-pre没有FDR分布，
不添加该项。分别记录 loss_wfdr、loss_wfdr_aux_N、loss_wfdr_dn_N，继续保留原 FGL/DDF 指标。
若手动与SDCP组合，复用其FGL边容差和非对称权重，避免W1重新引入容差内严格监督。
训练才计算该损失，不增加检测器推理计算。

交叉熵通过softmax对所有logits已有梯度，并非只有两个桶有梯度；W1新增的是显式非均匀距离约束。
q仍来自现有人工GT，不是已消除标注噪声的真值。抑制远处伪峰及提升AP75是待验证目标，
不能仅由损失形式保证所有多峰消失或检测增益。

## 配置、文件与固定协议

- `engine/deim/wasserstein_fdr.py`：独立W1实现、原插值权重接口、FP32及空匹配处理。
- `engine/deim/deim_criterion.py`：loss_local接入与 `use_wfdr/wfdr_weight` 开关。
- `tools/run_sar_module_experiment.sh`：记录开关、lambda_w和新损失外部权重。
- `tests/test_wfdr.py`：独立最优传输参考、解析梯度、分桶端点和实际检测训练路径验证。
- 单方法配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_wfdr_58e_b28_old_server.yml`。
- 组合配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pgd_rem_clcr_lite_wfdr_58e_b28_old_server.yml`。

两配置继承同一58轮baseline；组合配置额外开启PGD-REM及CLCR-Lite。
协议：DEIM-D-FINE-M / HGNetv2-B2、7类ID0–6、train3047 / val442 / test879，
seed0、GPU0/1、全局batch28（每卡14）、基础/验证640、训练多尺度480–800；
warmup432、flat29、no_aug8、lr_gamma0.5；增强policy [4,29,50]、mixup [4,29]、stop50。
AdamW主LR3.5e-4、骨干非BN LR3.5e-5、WD1e-4，Mosaic/MixUp、EMA、AMP FP16、
FP32 loss/retry及query×class展平top300后处理继承。两方法都从同一COCO M权重 `-t` 开始。
现有SAR-AIRcraft目录和0–6标注不变，无需重新准备数据。启动器归档配置、Git revision、seed、
batch、输入、优化器、增强、EMA/AMP、初始化和评估协议。

## 服务器更新及启动

若服务器工作树有改动或分叉，先检查并保留改动，不强制覆盖。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_wfdr_58e_b28_old_server.yml
```

启动器创建新目录，用 `CUDA_VISIBLE_DEVICES=0,1`、
`torchrun --standalone --nproc_per_node=2 train.py`；
执行 `nohup bash "$COMMAND" > "$LOG" 2>&1 < /dev/null &` 后，保存 `$!` 至launcher.pid再disown。
首次日志为console.log。前一实验结束后，组合实验另起新目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pgd_rem_clcr_lite_wfdr_58e_b28_old_server.yml
```

## 日志、指标、TensorBoard及恢复

重连后显式设置OUT为启动器打印的目录并激活环境：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际W-FDR运行目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C退出日志跟踪后，读取最佳验证AP和损失：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(line) for line in (Path(sys.argv[1]) / 'log.txt').read_text().splitlines() if line.startswith('{')]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
for label, row in [('best_val', best), ('last_logged', rows[-1])]:
    losses = {key:value for key,value in row.items() if key in ('train_loss_fgl', 'train_loss_ddf') or key.startswith('train_loss_wfdr')}
    print(label, 'epoch=', row['epoch'],
          'AP/AP50/AP75=', [round(value * 100, 4) for value in row['test_coco_eval_bbox'][:3]],
          'losses=', losses)
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

JSONL指标在log.txt；COCO bbox第0/1/2项是AP/AP50/AP75，百分数报告乘100。
TensorBoard在summary/，训练COCO数据在eval/latest.pth。
原进程停止后恢复同一W-FDR实验：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原W-FDR运行目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复用resolved_config.yml和 `-r last.pth`，不同时传 `-t`；日志为打印的resume_*.console.log。
本协议last.pth正常最后更新于epoch49，不是最后一轮权重。

## 按验证AP选择权重并评估测试集

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原W-FDR运行目录
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

按实际验证AP选择best_stg1.pth/best_stg2.pth，独立评估数据在 `$EVAL/eval.pth`，指标见控制台。

## 本地功能验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_wfdr.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sdcp.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_clcr_lite.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_clcr.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_query_prototypes.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_pgd_rem.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sar_58e_protocol.py -v
```

验证非均匀W1与独立最优传输计算的一致性、远近伪峰的解析梯度、双点目标、最后区间/端点、
原DFL插值权重复用、AMP/空匹配、IoU权重、原FGL/DDF不变、所有aux/DN路径及与SDCP容噪兼容。
结果：W-FDR 12项、SDCP 14项、Lite 11项、原CLCR 10项、原decoder/MPQ回归7项、PGD-REM 6项、
58轮协议5项，合计65项通过。启动器、本文6段Bash命令及嵌入Python片段语法检查通过，
未执行服务器命令。W1解析值与SciPy独立最优传输结果一致，远近伪峰的距离比例与解析梯度一致。
本地不包含真实数据训练和CUDA双卡吞吐/显存测试，不据此宣称检测收益。
