# SAR-AIRcraft：完整 BCCQ，旧服务器 102 轮实验

## 完整实现与实验定义

本组只在冻结 DEIM-D-FINE-M baseline 上增加完整 BCCQ，从相同 COCO-M 权重重新初始化训练。
不叠加 REM、PACFG、CLCR、SDCP、MPQ、SCS 或 OST-DA。
首轮同时启用六源证据、四层七类别查询、源与空间联合注意力、型号/质量分解监督。
方法来源、公式及贡献边界见 [完整研究规格](sar_band_conditioned_class_queries_research_2026-11-01.md)。

配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_bccq_102e_b28_old_server.yml`。

| 项目 | 设置 |
|---|---|
| 模型与数据 | DEIM-D-FINE-M / HGNetv2-B2；SAR-AIRcraft，7 类 ID0–6，train3047 / val442 / test879 |
| 初始化 | `/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth`，根目录 train.py 的 `-t`，新建优化器 |
| 轮数 / seed | 102 轮，日志 epoch0–101；seed0 |
| GPU / batch | GPU0、1；训练全局28，每卡14；验证全局64，每卡32 |
| 输入 | 训练多尺度480–800，基础/验证640；epoch90起固定640 |
| 优化器 | AdamW，主 LR3.5e-4，骨干非BN LR3.5e-5，WD1e-4，Norm/BN WD=0 |
| 调度 | flatcosine，warmup2000 steps，flat49，no_aug12，lr_gamma0.5 |
| 增强 | policy[4,49,90]，Mosaic/MixUp在[4,49)，PhotometricDistort在[4,90)，水平翻转；ZoomOut/IoUCrop关闭 |
| EMA / 精度 | EMA decay0.9999、warmups1000；SyncBN；AMP FP16、FP32 loss/retry；clip norm0.1 |
| BCCQ | d128，4头，FFN512，4层，7类查询；6源，每源5×5；query_chunk_size64，非重入激活重计算 |
| 评价 | 每轮 val、EMA、FP32、640、query×class展平top300；按最佳val AP选同一权重，统一报告test |

直接使用 `/home/zxy/sar/datasets/SAR-AIRcraft-1.0` 的原图与 `COCO/train.json`、`val.json`、`test.json`。
无需重新准备数据或修改标注。类别顺序为 A220、A320/321、A330、ARJ21、Boeing737、Boeing787、other。
目标是相对同协议102轮 baseline 的 COCO AP@[.50:.95] 增加1.5–2点，目前没有训练收益结论。

## 实现范围

- `engine/deim/band_conditioned_queries.py`：P3/P4空间与P3的LL/LH/HL/HH；保持子带符号；实例内七类查询；独立源路由与源内空间注意力。
- `engine/deim/dfine_decoder.py`：四层逐层更新、每层联合分数、普通/aux/DN字段拆分与PyTorch部署裁剪；原FDR与LQE统计保留。
- `engine/deim/deim_criterion.py`：正匹配型号CE与标量质量损失；分类使用各层原匹配，原框/FGL匹配并集不变；记录按类别的门控统计。
- `engine/solver/det_engine.py`：将BCCQ训练诊断写入JSONL与TensorBoard。
- `tools/run_sar_module_experiment.sh`：记录完整方法元数据；默认使用旧服务器路径，显式环境变量仍可覆盖。
- `tests/test_bccq.py`、`tests/check_bccq_ddp.py`：公式、完整检测器联合反向、AMP、DN、重计算、EMA/恢复/部署及两进程梯度验证。

型号CE只监督原匹配正样本，不按IoU缩小类别目标。质量目标为detached IoU^1.5，负样本为0；
质量项正样本权重1，负样本权重Q^1.5（当前mal_alpha未指定，系数为1）。两项权重均为1。
四层普通/aux/DN使用分解损失，DN负样本只有质量项；encoder/pre-output/DN pre-output沿用原MAL。
最终分数 `sigmoid(objectness+LQE) * softmax(type)` 以稳定FP32 logit形式进入原matcher和后处理。
原L1/GIoU/FGL/DDF及其权重保留，DDF读取最终联合分数作为teacher质量。

新增参数正常初始化，从epoch0参与训练。新增分支在隔离随机状态中构建，保留共同权重及后续DN随机序列。
只保留旧分类头0用于pre-output，其余三个不再使用的旧分类头删除。
BCCQ分支1,341,956参数，整网净增1,336,559个可训练参数。
查询分块与激活重计算均不删减任何证据源、类别或层。

## 服务器 Git 更新与正式启动

实现分支为 `codex/bccq-102e`，基于main的8eff55a。先查看服务器工作树；有改动或分叉时保留并处理，
不使用reset或强制checkout。以下命令在源代码已推送后执行：

```bash
cd /home/zxy/sar/repos/deim
git status --short --branch
git fetch origin
git switch codex/bccq-102e
git pull --ff-only origin codex/bccq-102e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_bccq_102e_b28_old_server.yml
```

启动器创建时间戳/随机后缀的新目录，保存resolved_config、Git revision、seed、batch、输入、LR、增强、
EMA/AMP、初始化权重和评估协议。实际使用 `torchrun --standalone --nproc_per_node=2 train.py`。
后台执行 `nohup bash "$COMMAND" > "$LOG" 2>&1 < /dev/null &`，先保存$!到launcher.pid，再disown。
两卡同时空闲才启动；不排队、不停止已有运行。以打印的STARTED、OUT、PID及随后console.log确认实际状态。

## 重连后查看进程、日志与指标

每次重连都重新激活环境，并显式设置本次启动器打印的OUT：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为本次BCCQ实际目录
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
nvidia-smi -i 0,1
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

退出tail后，可持续查看GPU与工作进程：

```bash
watch -n 2 nvidia-smi -i 0,1
pgrep -af 'torchrun|train.py'
```

输出最佳验证与最近一轮的百分数指标及训练门控统计：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

rows = [json.loads(s) for s in (Path(sys.argv[1]) / 'log.txt').read_text().splitlines() if s.strip()]
by_epoch = {r['epoch']: r for r in rows if 'test_coco_eval_bbox' in r}
rows = [by_epoch[e] for e in sorted(by_epoch)]
best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
for label, row in [('best_val', best), ('last_logged', rows[-1])]:
    print(label, 'epoch=', row['epoch'], 'AP/AP50/AP75/APs/APm/APl=',
          [round(x * 100, 4) for x in row['test_coco_eval_bbox'][:6]],
          'type_loss=', row.get('train_loss_bccq_type'),
          'quality_loss=', row.get('train_loss_bccq_quality'))
names = ['A220', 'A320/321', 'A330', 'ARJ21', 'Boeing737', 'Boeing787', 'other']
for category, name in enumerate(names):
    count = best.get(f'train_bccq_positive_c{category}', 0)
    print(name, 'mean_positive_per_batch=', best.get(f'train_bccq_positive_c{category}'),
          'source_weights=', {s: best.get(f'train_bccq_gate_sum_c{category}_{s}', 0) / count if count else None
                             for s in ['p3', 'p4', 'll', 'lh', 'hl', 'hh']},
          'gate_entropy=', best.get(f'train_bccq_gate_entropy_sum_c{category}', 0) / count if count else None)
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

门控统计来自训练期原匹配及对应GT类别，跨rank汇总后按epoch平均；不是验证集EMA的注意力统计。
`train_bccq_positive_cN`是每batch正匹配数的epoch均值，不是唯一GT计数。
日志中的gate_sum/entropy_sum是每batch总和的epoch均值；除以同epoch的positive均值，
得到按实例加权的源权重/熵，避免稀有类别缺席的batch把均值压低。
JSONL为log.txt，TensorBoard为summary/，COCO数据为eval/latest.pth。
COCO bbox索引0/1/2为AP@[.50:.95]/AP50/AP75，百分数乘100。

## 中断后的恢复

原进程结束后从原目录恢复，保持BCCQ的模型、优化器和EMA状态：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原BCCQ实际目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
tail -f "$(ls -t "$OUT"/resume_*.console.log | head -n 1)"
```

恢复使用原resolved_config.yml和`-r last.pth`，不同时传`-t`。
控制台日志为新的resume_*.console.log。当前solver仅在epoch<90保存last.pth，正常最后更新于epoch89，
不是第101轮最终权重。最后12轮中断时，恢复命令会从该恢复点重跑收尾。

## 按验证AP选择权重，评估test

按最高验证AP对应阶段选择best_stg1/best_stg2；不直接认定某个阶段更优。
训练完成后执行：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为本次BCCQ实际目录
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

测试指标见console.log，评估COCO数据为EVAL/eval.pth。恢复过的运行应同时核对实际候选权重的验证AP。
同一checkpoint报告AP/AP50/AP75和逐类指标，不拼接不同轮次的峰值。

## 本地验证与范围

使用Windows Conda pytorch，定向验证包括：带符号Haar与奇数尺寸坐标、六源路由方程、
联合概率极端值、CE/质量损失公式、空正样本、实例/框detach和视觉梯度、完整四层普通/aux/DN路径、
分块与重计算等价、共同初始化/RNG、BF16 AMP、真实DEIM-M骨干/FPN/BCCQ联合反向、EMA/优化器恢复及部署输出一致性。
真实配置构建还核对102轮继承协议，除完整方法与输出元数据外保持baseline设置。

两进程gloo DDP验证连续两次更新，其中一rank没有GT，开启重计算且find_unused_parameters=False；
全部127个BCCQ参数张量的梯度与全局batch参考一致。
本地结果：BCCQ 16项、CLCR-Lite 11项、SDCP 14项、W-FDR 12项、现有模块接入8项，共61项通过。
启动器元数据生成已用实际102轮配置验证，包含六源、四层、七类、质量目标与损失权重。

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_bccq.py -v
conda run --no-capture-output -n pytorch python tests/check_bccq_ddp.py
```

本地未执行真实SAR数据训练、CUDA FP16吞吐/峰值显存或ONNX/TensorRT验证。
开发环境不能直连服务器，102轮训练需在旧服务器执行启动命令；本文不宣称训练已经启动或涨点已实现。
