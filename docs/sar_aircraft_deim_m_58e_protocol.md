# SAR-AIRcraft DEIM-M 58 轮短程协议（规格，交 Codex 实现）

状态：规格已定，待实现和启动。旧 102 轮配置、结果和文档保持不变；
从本协议开始，新的 baseline 和所有新方法都以 58 轮配置为父配置。

## 1. 为什么改

依据见 [研究诊断](research_diagnosis_2026-10-28.md) 第 5.4 节（baseline 各 checkpoint 的 test 评估）：

- test AP 在 102 轮协议的 epoch 59 左右进入平台（45.9），之后到 epoch 90 波动不超过 0.3，后约 40 轮没有收益。
- 2000 步平方 warmup 在本数据集上约等于 18.5 轮（每轮 108 步）。官方 COCO 上同样 2000 步只相当于约 0.55 轮。
  前 20 轮 AP 接近 0，大部分是 warmup 造成的。
- DEIM 官方 `configs/base/deim.yml` 本身就有 58 轮短程调度（policy [4,29,50]、flat 29、no_aug 8），
  官方 L/X 在 COCO 上就用这一调度。本协议直接沿用官方阶段划分，只把 warmup 改为按轮数计。

## 2. 参数变更（只改调度相关项）

| 参数 | 102 轮协议 | 58 轮协议 | 依据 |
|---|---|---|---|
| `epoches` | 102 | **58** | 官方短程：50 轮主训练 + 8 轮收尾 |
| `flat_epoch` | 49 | **29** | 官方公式 4 + 50/2 |
| `no_aug_epoch` | 12 | **8** | 官方 base |
| `warmup_iter` | 2000 | **432** | 4 轮 × 108 步：warmup 正好在强增强开始（epoch 4）时结束 |
| `train_dataloader.dataset.transforms.policy.epoch` | [4,49,90] | **[4,29,50]** | 官方 base |
| `train_dataloader.collate_fn.mixup_epochs` | [4,49] | **[4,29]** | 官方 base |
| `train_dataloader.collate_fn.stop_epoch` | 90 | **50** | 官方 base；多尺度、强增强在此停止，同时切换到第二阶段 |

保持不变：
- 模型：DEIM-D-FINE-M，HGNetv2-B2，4 层 decoder，7 类，COCO M 权重 `-t` 初始化。
- 数据和划分：官方 train 3047 / val 442 / test 879，原 COCO 标注。
- 优化器：AdamW，主 LR 3.5e-4，backbone 非 BN 3.5e-5，WD 1e-4，Norm/BN WD 0，betas (0.9, 0.999)，`lr_gamma` 0.5。
- 增强组成：关闭 RandomZoomOut / RandomIoUCrop；Mosaic 外层 p 0.5，MixUp p 0.5，PhotometricDistort p 0.5，水平翻转，
  480–800 多尺度（`base_size_repeat` 6），基础和验证尺寸 640。只是各阶段的起止轮按上表平移。
- 全局 batch 28（每卡 14），GPU 0、1，seed 0，AMP FP16、FP32 loss、非有限 batch 用 FP32 重算，SyncBN，clip 0.1。
- EMA：decay 0.9999，warmups 1000，第二阶段的 restart decay 0.9999，**不改**。
  当前实现的有效衰减是 `0.9999·(1−e^(−n/1000))`，会随步数自适应：
  58 轮协议在 stop_epoch 时 EMA 质量的半衰窗口约为最近 139 步（约 1.3 轮），不存在早期权重拖尾，没有理由偏离官方值。
- `checkpoint_freq` 4，验证 batch 64，后处理 top-300，不额外加 NMS。

## 3. 换算后的步数（每轮 108 步，drop_last）

总步数 6264；warmup 0–432（平方上升）；平台 LR 432–3132；余弦 3132–5400 下降到 0.5 倍；
5400–6264（epoch 50–57）保持最小 LR。epoch 0–3 不开强增强，4–28 开 Mosaic / MixUp，
4–49 开光度扰动和多尺度，50–57 固定 640 收尾。

solver 行为（已核对 `engine/solver/det_solver.py`）：epoch == 50 时回载 `best_stg1.pth`，并重置 EMA decay；
`last.pth` 只在 epoch < 50 时更新（正常最后一次是 epoch 49），不是最终权重；
第二阶段最佳保存为 `best_stg2.pth`。阶段内每 4 轮保存一个 `checkpoint00XX.pth`（3, 7, …, 47）。

## 4. 评估口径（与旧协议的区别）

1. 训练时每轮在 val 上评估，按 val AP 在 `best_stg1` / `best_stg2` 中选最终权重（官方划分不变）。
2. **模块比较与论文报告以 test 为准**：对选出的权重做一次 test 评估，报告 AP / AP50 / AP75 / AR100 和各类 AP。
3. 曲线诊断：对 `checkpoint0011…0047`、`best_stg1`、`best_stg2` 都做 test 评估，记录 test AP 随 epoch 的变化。
   只用于检查收敛，不用来选权重。
4. 类别竞争重打分 `s·sqrt(s_c/Σs)` 是推理组件，不属于 baseline，单独报告。

## 5. 验收与回退

参照：旧 102 轮 baseline 的 val 选择权重 test AP 45.7（`train.py --test-only` 口径），
平台期 checkpoint 为 45.8–46.0。

- 新 baseline test AP ≥ 45.2（与旧值相差 0.5 以内）→ 采纳 58 轮作为后续所有实验的父协议。
- test AP < 45.2 → 改用 72 轮回退方案并重跑：`epoches 72`、`flat_epoch 34`、`no_aug_epoch 12`、
  policy [4,34,60]、mixup [4,34]、`stop_epoch 60`、`warmup_iter 432`（官方公式 4 + 60/2）。
  不要在两者之间再做搜索。

## 6. 给 Codex 的实现任务

1. 新建 `configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_58e_b28.yml`：
   - 单父继承 `./deim_hgnetv2_m_sar_aircraft_102e_b28.yml`，只覆盖第 2 节表中 7 个键和 `output_dir`；
   - `output_dir: /root/zxy/experiments/deim_m_sar_aircraft_58e_b28/seed0`；
   - 加 `experiment_protocol_document: docs/sar_aircraft_deim_m_58e_protocol.md`，让启动器归档本文。
   - 注意 YAML 合并时列表是整体替换，`policy.ops` 继承自父配置，保持 ['Mosaic','RandomPhotometricDistort']。
2. 新增定向回归 `tests/test_sar_58e_protocol.py`（CPU，只解析配置，不训练）：
   - 解析后上述 7 个键等于规格值；
   - 与 102 轮配置相比，除这 7 个键和 `output_dir` / `experiment_protocol_document` 外，其余解析结果完全相同；
   - 用 `flat_cosine_schedule` 计算 iter_per_epoch=108 时的边界：
     step 432 的 LR 为 3.5e-4，step 3132 为 3.5e-4，step 5400 及以后为 1.75e-4。
3. 不修改 solver、模型、损失、数据、旧配置或旧文档。不做哈希校验或冒烟测试。
4. 提交并 push 到 `origin/main`；在本文末尾追加"实现记录"（commit、测试结果）。

## 7. 服务器命令

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
```

已有两个未跟踪的 OGSOD 配置与本任务无关，保留即可。工作区有修改或分叉时先检查，不要 reset。

启动（启动器会自己检查两卡空闲、建新目录、保存解析配置 / revision / 协议，并以 nohup 和 disown 启动）：

```bash
bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_58e_b28.yml
```

把启动器打印出的实际目录设为 OUT 后，监控：

```bash
OUT=/root/zxy/experiments/替换为实际目录
tail -f "$OUT/console.log"
tail -n 3 "$OUT/log.txt"
nvidia-smi -i 0,1
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

只在进程已停止时恢复：

```bash
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

训练结束后，先按 val 选权重：

```bash
python - "$OUT/log.txt" <<'PY'
import json,sys
r=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
b=max(r,key=lambda x:x['test_coco_eval_bbox'][0])
print('best val epoch',b['epoch'],'AP',round(b['test_coco_eval_bbox'][0]*100,2),'-> best_stg2' if b['epoch']>=50 else '-> best_stg1')
PY
```

然后对所选权重和曲线 checkpoint 做 test 评估（两卡，按顺序执行）：

```bash
ulimit -Sn 65535
for K in best_stg1 best_stg2 checkpoint0011 checkpoint0015 checkpoint0019 checkpoint0023 checkpoint0027 checkpoint0031 checkpoint0035 checkpoint0039 checkpoint0043 checkpoint0047; do
  EVAL="$OUT/test_eval/$K"; mkdir -p "$EVAL"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
    -c "$OUT/resolved_config.yml" -r "$OUT/$K.pth" --test-only \
    --output-dir "$EVAL" --summary-dir "$EVAL/summary" \
    -u val_dataloader.dataset.ann_file=/root/zxy/datasets/SAR-AIRcraft-1.0/COCO/test.json \
    > "$EVAL/console.log" 2>&1
done
grep -H -A2 "IoU=0.50:0.95 | area=   all | maxDets=100 ]" "$OUT"/test_eval/*/console.log
```

`test_coco_eval_bbox` 的第 0/1/2 项是 AP / AP50 / AP75，取值 0..1，报告时乘 100。

## 8. 后续（不在本任务内）

58 轮 baseline 通过验收后，SCS 以 58 轮配置为父配置建子配置重跑。旧的 SCS 配置继承自 102 轮，不能直接复用其结果做对比。

## 9. 实现记录（2026-10-28）

状态更新：已实现并完成本地 CPU 定向回归；实验未启动，尚无 58 轮训练或 test 指标。

- 实现 commit：`93df36b`（`main`，Add SAR-AIRcraft 58-epoch baseline protocol and regression tests）。
  本节作为后续文档提交补记，避免在实现提交中循环引用自身 revision。
- 新增 `configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_58e_b28.yml`：仅继承 102 轮配置，
  覆盖七项调度值、`output_dir` 和协议归档路径。`policy.ops` 继承为
  `['Mosaic', 'RandomPhotometricDistort']`；其余解析配置与 102 轮基线一致。
- 新增 `tests/test_sar_58e_protocol.py`：四项回归覆盖单父继承与归档路径、七项调度与
  `policy.ops`、其余解析配置全量一致性，以及 `flat_cosine_schedule` 的 warmup / flat /
  no-augmentation 边界。比较时只排除七项调度、两个归档/输出字段和非运行参数 `__include__`。
- Windows Conda `pytorch` 环境运行结果：**4 tests passed，OK**，测试主体耗时 0.129 秒。
  step 432 / 3132 为 `3.5e-4`，step 5400 / 5401 / 6263 / 6264 为 `1.75e-4`；
  同时核对 step 0 和平方 warmup 中点 step 216。
- 验证仅解析配置和计算 LR，不实例化检测器、不加载数据或权重、不启动训练；
  未进行哈希校验或冒烟测试。未修改 solver、模型、损失、数据、旧配置或旧协议。
- 现有 `tools/run_sar_module_experiment.sh` 已读取 `experiment_protocol_document` 并归档为
  `method_and_protocol.md`，无需修改启动器。本次沿用本文明确指定的 `/root/zxy/...` 路径。
  数据继续使用原 COCO 标注，无需重新准备。

本地复现命令（PowerShell）：

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sar_58e_protocol.py -v
```

服务器更新、两卡启动和恢复命令见第 7 节，均仅交接，未远程执行。
启动器使用新建目录，保存解析配置、Git revision、seed、全局 batch、输入尺寸、
优化器与调度、增强、EMA、AMP、初始化权重和评估口径，并以
`nohup ... > console.log 2>&1 < /dev/null &` 启动，保存 `$!` 后 `disown`。
服务器工作区有修改或与远端分叉时先检查，保留已有 OGSOD 配置。

重连后监控和查看逐轮指标（先将 `OUT` 改为启动器打印的实际目录；恢复训练时也显式重设）：

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
export OUT=/root/zxy/experiments/替换为实际目录
tail -f "$OUT/console.log"
# 恢复时启动器打印新的 resume_*.console.log，改为跟踪该日志。
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
nvidia-smi -i 0,1
tail -n 3 "$OUT/log.txt"
python - "$OUT/log.txt" <<'PY'
import json, sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.startswith('{')]
for row in rows[-3:]:
    values = row['test_coco_eval_bbox']
    print('val epoch', row['epoch'], dict(zip(['AP', 'AP50', 'AP75', 'AR100'],
          [round(values[i] * 100, 2) for i in [0, 1, 2, 8]])))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

`last.pth` 正常最后为 epoch 49，用于已停止运行的恢复；它不是最终权重。
`best_stg2.pth` 仅在实际保存时存在，第 7 节评估循环可在 `EVAL=...` 前加入
`[ -f "$OUT/$K.pth" ] || continue` 跳过不存在的阶段权重。
最终权重按 val AP 选择，曲线 checkpoint 的 test AP 仅用于诊断；不要按 test AP 选权重。

test 评估完成后，从 `eval.pth` 提取 AP / AP50 / AP75 / AR100 与各类 AP（原始值乘 100）：

```bash
python - "$OUT" /root/zxy/datasets/SAR-AIRcraft-1.0/COCO/test.json <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import torch

names = {row['id']: row['name'] for row in json.loads(Path(sys.argv[2]).read_text())['categories']}
def percent(values):
    valid = values[values > -1]
    return round(float(valid.mean()) * 100, 2) if valid.size else None

for path in sorted((Path(sys.argv[1]) / 'test_eval').glob('*/eval.pth')):
    result = torch.load(path, map_location='cpu', weights_only=False)
    params = result['params']
    area, max_dets = params.areaRngLbl.index('all'), params.maxDets.index(100)
    precision = result['precision'][:, :, :, area, max_dets]
    recall = result['recall'][:, :, area, max_dets]
    metrics = dict(AP=percent(precision),
                   AP50=percent(precision[np.isclose(params.iouThrs, .5)]),
                   AP75=percent(precision[np.isclose(params.iouThrs, .75)]),
                   AR100=percent(recall))
    metrics['per_class_AP'] = {names[cat_id]: percent(precision[:, :, k])
                               for k, cat_id in enumerate(params.catIds)}
    print(path.parent.name, json.dumps(metrics, ensure_ascii=False))
PY
```
