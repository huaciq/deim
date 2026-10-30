# SAR-AIRcraft CLCR-Lite 实现与实验交接

## 已确认的实现选择

原 CLCR 及其已报告配置完整保留。新增 `DFINETransformer.use_clcr_lite`，与 `use_clcr`
互斥；新增配置显式关闭原 CLCR。Lite 只读取当前分类 logits、原 LQE 四边统计和当前预测框，
不创建局部 MLP、采样网格、局部特征图或 `grid_sample` 分支。
检测器本身的 deformable attention 仍有采样；若组合 SDCP，SDCP 自己的框内采样仍保留。

用户已确认以下三项规格澄清：

1. 采用 3 维竞争 + 2 维边缘摘要 + 3 维尺度，共 8 维；原 LQE 继续接收完整 20 维。
2. 仅输出层权重和偏置零初始化，第一层正常初始化，beta=1。两层权重全零会使隐藏层无法有效学习；
   输出层和 beta 同时为零也会使整个残差失去启动梯度。
3. 四个 decoder 层共享同一 Lite 头及 beta，整个模型仅新增 162 个可训练参数。

对每个 Query、每个类别，8 维输入固定按下列顺序排列：

| 维度 | 统计量 |
| --- | --- |
| 0 | 当前类概率 p_c=sigmoid(z_c) |
| 1 | p_c−max_{k≠c}(p_k) |
| 2 | H=−sum(p_norm log p_norm)/log(C)，C=7 |
| 3 | 四条边 top-k 概率均值的平均值 |
| 4 | 四条边 top-k 概率均值的总体标准差（unbiased=False） |
| 5 | ln(w) |
| 6 | ln(h) |
| 7 | ln(w/h)=ln(w)−ln(h) |

每边集中度直接取原 LQE 统计中已有的 top-k 均值，k=4，不添加学习参数。
宽高使用当前预测框相对于输入图像的归一化宽高，最小值截为 1e-6 后取对数。
因此尺度先验表达目标占图像的比例；ln(w*h) 可由维度 5+6 精确表示。
竞争统计使用校准前 logits。FP32 计算统计量，熵使用 log-sigmoid/log-softmax 保持极端 logits 稳定，
MLP 沿用 AMP 设置，修正返回原 logits dtype。

`MLP_tiny = Linear(8,16) → SiLU → Linear(16,1)`，七个类别共享 MLP 权重，
但各类别的概率、margin 不同，获得各自的修正。
最终为 `z_hat = z + original_LQE(edge20) + beta*MLP_tiny(input8)`。
MLP 共 161 参数，加 beta 共 162；原四层 CLCR 共 146,248 参数，新增校准参数减少约 99.89%。
推理只运行 eval_idx 输出层校准，部署裁剪仍保留同一个 Lite 头。

初始残差严格为零。第一步输出层能获得有效梯度；输出层更新后，第一层、beta 和输入统计路径
获得有效梯度。不 detach logits、边缘统计或框宽高。训练普通 Query 与 DN Query、最终层与 aux
均经过 Lite；pre-output 和 encoder 候选选择不经过 LQE，沿用原规则。
新头在 baseline 初始化完成后、隔离随机状态创建，共同权重及后续 DN 随机序列保持一致。
不增加辅助损失，不更改匹配或后处理规则；参数共享在 EMA、恢复和部署过程中保留。
共享头只在 Decoder 中注册一次，再传给各层 LQE，避免 state_dict 出现多份别名而使 EMA 重复更新。

AP_L 恢复至 0.582+、保持 AP75=0.733、改善 40 轮后过拟合是实验目标，不能由删去采样或减少参数
直接保证。现有信息也不能单独证明大目标下降由 grid_sample 导致。应比较同协议 baseline、原 CLCR
和 Lite 的最佳验证 AP、AP75、AP_L、训练曲线，再评估独立测试集。

## 文件与固定协议

- `engine/deim/dfine_decoder.py`：新增 CLCRLite，接入 LQE、当前框与共享头，原 CLCR 保留。
- `engine/solver/det_engine.py`：记录共享参数 `train_clcr_lite_beta`。
- `tools/run_sar_module_experiment.sh`：归档 Lite 开关、8 维定义、共享和初始化方式。
- `tests/test_clcr_lite.py`、`tests/check_clcr_lite_ddp.py`：统计、梯度、无局部采样、兼容性及共享参数验证。
- 单模块配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_clcr_lite_58e_b28_old_server.yml`。
- 两模块配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pgd_rem_clcr_lite_58e_b28_old_server.yml`。
- 三模块配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pgd_rem_clcr_lite_sdcp_58e_b28_old_server.yml`。

继承同一 58 轮协议：DEIM-D-FINE-M / HGNetv2-B2、7类 ID0–6、train3047 / val442 / test879，
seed0、GPU0/1、全局 batch28（每卡14）、基础/验证640、训练多尺度480–800；
warmup432、flat29、no_aug8、lr_gamma0.5，增强 policy [4,29,50]、mixup [4,29]、stop50。
AdamW 主 LR3.5e-4、骨干非 BN LR3.5e-5、WD1e-4，EMA、AMP FP16、FP32 loss/retry、
Mosaic/MixUp 和 query×class 展平 top300 后处理继承。三模块配置额外继承 SDCP 的分类及损失设定。
所有新实验从同一 COCO M 权重 `-t` 开始，不用旧 CLCR 的 optimizer 状态恢复 Lite。
现有 SAR-AIRcraft 路径及 0–6 标注不变，无需重新准备数据。启动器归档配置、Git revision、seed、
batch、输入、优化器、增强、EMA/AMP、初始化路径和评估协议。

## 服务器更新与启动

服务器工作树若有改动或分叉，先检查并保留改动，不强制覆盖。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_clcr_lite_58e_b28_old_server.yml
```

启动器创建新目录，用 GPU0/1、`torchrun --standalone --nproc_per_node=2 train.py`，
以 `nohup bash "$COMMAND" > "$LOG" 2>&1 < /dev/null &` 启动，保存 `$!` 至 launcher.pid 后 disown。
日志为 console.log，不覆盖已有运行。

前一实验结束后，组合实验使用另一配置另起新目录；两种配置选一个运行：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
CONFIG=configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pgd_rem_clcr_lite_58e_b28_old_server.yml
# 三模块时改为：
# CONFIG=configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pgd_rem_clcr_lite_sdcp_58e_b28_old_server.yml
bash tools/run_sar_module_experiment.sh "$CONFIG"
```

## 日志、指标、恢复

重连后显式设置 OUT 为启动器打印的实际目录并激活环境：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际Lite运行目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C 退出日志跟踪后，查看最佳验证和最近指标；AP_L 是 COCO bbox 第5项：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(line) for line in (Path(sys.argv[1]) / 'log.txt').read_text().splitlines() if line.startswith('{')]
best = max(rows, key=lambda row: row['test_coco_eval_bbox'][0])
for label, row in [('best_val', best), ('last_logged', rows[-1])]:
    metric = row['test_coco_eval_bbox']
    print(label, 'epoch=', row['epoch'],
          'AP/AP50/AP75/AP_L=', [round(metric[i] * 100, 4) for i in [0, 1, 2, 5]],
          'beta=', row.get('train_clcr_lite_beta'), 'pgd_alpha=', row.get('train_pgd_rem_alpha'))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

训练 JSONL 在 log.txt，AP/AP50/AP75 分别为 test_coco_eval_bbox 第0/1/2项，百分数报告乘100；
TensorBoard 在 summary/，COCO 数据在 eval/latest.pth。原进程停止后恢复同一 Lite 运行：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原Lite运行目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复用原 resolved_config.yml 和 `-r last.pth`，不同时传 `-t`；日志为打印的 resume_*.console.log。
当前协议 last.pth 正常最后更新于 epoch49，不是最终轮权重。

## 选择最佳验证权重并评估测试集

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原Lite运行目录
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

按实际验证 AP 选择 best_stg1.pth/best_stg2.pth；独立评估指标见控制台，COCO 数据在 eval.pth。

## 本地功能验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_clcr_lite.py -v
conda run --no-capture-output -n pytorch python tests/check_clcr_lite_ddp.py
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_clcr.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sdcp.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_query_prototypes.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_pgd_rem.py -v
```

验证范围：8维定义、极端数值、AMP梯度启动、四层共享参数及优化器去重、无局部采样、原LQE保留、
普通/aux/DN、PGD-REM/SDCP组合、恢复及部署；CPU双进程比较共享头梯度与串行全局均值。
结果：Lite 11项、原CLCR 10项、SDCP 14项、原decoder/MPQ回归7项、PGD-REM 6项，合计48项通过；
CPU双进程连续两步的共享头梯度比较通过，EMA一次更新的解析值验证通过。启动器、本文6段Bash命令
及嵌入Python片段语法检查通过，没有执行服务器命令。
本地验证不包含真实数据训练及CUDA双卡吞吐/显存评估，不据此宣称AP或过拟合改善。
