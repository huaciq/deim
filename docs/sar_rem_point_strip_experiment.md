# SAR-AIRcraft REM：Point-Strip Dual-Path，58轮实验

## 方法与插入位置

新变体为 `HybridEncoder.rem_variant: point_strip`，仍通过 `use_rem: True` 开启。
插入位置是stride-8的256通道input_proj之后、FPN/PAN融合之前。
一次Haar DWT得到LL/LH/HL/HH，四路处理后使用一次IDWT和原共享alpha残差。

本组只替换LH/HL的内部结构，采用全256通道方案：

```text
P_LH = Conv1x1 + BN + SiLU (LH)
F_LH = P_LH + beta_LH * [Conv1x3 + BN + SiLU](P_LH)

P_HL = Conv1x1 + BN + SiLU (HL)
F_HL = P_HL + beta_HL * [Conv3x1 + BN + SiLU](P_HL)

Y = X + alpha * IDWT(LL_original_branch, F_LH, F_HL, HH_original_branch)
```

LH取消原来的后续3x1和3x3；HL取消后续1x3和3x3。
并联求和后没有额外激活或空间卷积；点路径直接连接到IDWT输入。
LL仍为1x1 Conv/BN/SiLU，HH仍为1x1→3x3 Conv/BN/SiLU。
不拆分128+128通道，不增加局部采样、校准头或额外损失。

两个beta独立、可学习、初始化为1。外层共享alpha仍为0，完整模块初始为Y=X。
beta=1让条带路在alpha开始学习后即有梯度通路，避免内外缩放同时从0起步。
点路径复用原REM的入口1x1，条带路复用原来的第一条方向卷积，保留这些权重的初始化。
LL/HH及模块构建后的随机状态与原REM相同，删除的四个卷积块不保留为参数。

这里的“点保真”是1x1投影之后的直接空间旁路，不代表原始子带数值无损；
1x1、BN和SiLU仍会改变通道表示。原LH/HL的卷积支持均为5x5，也不能据此证明方向信息完全消失。
本组检验缩小跨轴混合是否有利于紧凑结构与类别判别；“去混叠”按用户公式落实为移除串行跨轴扩散，
保留原Haar分析/合成滤波器，并未加入新的抗混叠低通算子。
对A220的恢复、A330的保留以及背景误检抑制，均以实验结果检验。

## 参数与计算

| 适配器（C=256） | 可训练参数 | 卷积参数 | 640输入时适配器卷积GMACs |
|---|---:|---:|---:|
| 原REM | 2,823,681 | 2,818,048 | 4.508877 |
| REM-point-strip | 1,248,771 | 1,245,184 | 1.992294 |

适配器可训练参数减少1,574,910，约55.8%。新增的两个beta已计入。
卷积MAC按80x80的stride-8特征、40x40子带计算，仅统计适配器卷积，
不包含变换、BN、激活和残差；该数值不代表整模型计算量或实测GPU时延。

## 固定的58轮协议

新配置单父级继承原REM的old_server 58轮配置，仅新增variant并使用独立输出/文档路径。
原baseline、REM、per_band和其他已报告变体均保留。

- DEIM-D-FINE-M / HGNetv2-B2，7类ID0–6；沿用train3047、val442、test879及原标注。
- seed0，58轮，全局batch28，双卡每卡14；固定评估640，训练多尺度480–800。
- warmup432，flat29，no_aug8，lr_gamma0.5。
- 增强policy [4,29,50]，mixup [4,29]，stop_epoch50。
- RandomZoomOut、RandomIoUCrop全程关闭，其余增强沿用原REM。
- AdamW，主LR3.5e-4，骨干非BN LR3.5e-5，WD1e-4；两个beta沿用普通适配器参数的优化设置。
- EMA、SyncBN、AMP FP16、FP32 loss及原有FP32 retry策略。
- 原4层Decoder、采样点[3,6,3]、reg_max32、MAL/框/FGL/DDF、匹配与top300后处理。
- 从同一COCO-M权重通过-t初始化、使用新优化器，从第0轮训练；不是继续训练原REM权重。

数据直接使用原old_server配置的 `/home/zxy/sar/datasets/SAR-AIRcraft-1.0`，无需数据准备。
主要比较对象为同协议58轮baseline和原REM；同时记录AP/AP75/APm/APl与逐类AP。
重点观察A220是否恢复，A330与其他类是否保留，以及后期均值是否改善。

## 文件

- `engine/deim/radar_enhancement.py`：PointStripDualPath与point_strip选项。
- `engine/solver/det_engine.py`：记录共享alpha、LH/HL beta。
- `tools/run_sar_module_experiment.sh`：归档点路、条带核、beta初始化及完整协议元数据。
- `configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_point_strip_58e_b28_old_server.yml`：独立58轮配置。
- `tests/test_rem_point_strip.py`：解析空间支持、梯度、工厂配置与EMA/保存恢复验证。

HybridEncoder已有rem_variant接口，直接传递新选项；网络插入位置和生产默认值不变。

## 服务器Git更新与启动

检查工作树，保留服务器已有改动；若存在分叉先处理，更新采用ff-only。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_point_strip_58e_b28_old_server.yml
```

启动器新建时间戳/随机后缀目录，归档配置、Git revision、seed、全局batch、输入、LR、轮数、
增强、EMA、AMP、初始化和评估协议，使用两卡0、1运行
`torchrun --standalone --nproc_per_node=2 train.py`。
现有启动器仅在两卡同时空闲时启动；不排队、不停止其他任务。
nohup后台运行，stdout/stderr写console.log、stdin来自/dev/null；先保存$!到launcher.pid再disown。
根据打印的STARTED、OUT、PID和随后训练日志确认启动情况。

## 日志、指标、进程和TensorBoard

重连先激活环境、显式设置OUT为启动器打印的实验目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为本次point_strip实验目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

退出tail后读取最佳AP对应同一轮的全部指标和训练缩放参数均值：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
rows = [json.loads(line) for line in (run / 'log.txt').read_text().splitlines() if line.strip()]
by_epoch = {r['epoch']: r for r in rows if 'test_coco_eval_bbox' in r}
best = max(by_epoch.values(), key=lambda r: r['test_coco_eval_bbox'][0])
latest = by_epoch[max(by_epoch)]
for label, row in [('best_AP', best), ('last_logged', latest)]:
    print(label, 'epoch=', row['epoch'], 'AP/AP50/AP75/APs/APm/APl=',
          [round(v * 100, 4) for v in row['test_coco_eval_bbox'][:6]],
          'training_scales=', {k:v for k,v in row.items() if k.startswith(('train_rem_alpha', 'train_rem_beta'))})
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

log.txt为JSONL，summary/保存TensorBoard，eval/latest.pth保存训练COCO评估数据。
新增日志为train_rem_beta_lh和train_rem_beta_hl，原train_rem_alpha保留；
TensorBoard对应Module/rem_beta_lh、Module/rem_beta_hl、Module/rem_alpha。
这些是训练参数的epoch均值，不是EMA checkpoint中的最终参数值。
COCO bbox指标第0/1/2项为AP/AP50/AP75，日志中的0–1数值乘100报告为百分数。

## 中断恢复

原训练进程停止后，使用保存的resolved_config.yml与-r恢复：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为待恢复point_strip目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复仍使用GPU0、1，后台nohup与PID记录同启动流程，输出为打印的resume_*.console.log。
不同时传-t。last.pth只在stop_epoch之前更新，58轮正常最后保存于epoch49，
不是最终第58轮权重；收尾阶段中断会从该保存点重新走后续阶段。

## 选择最佳checkpoint并独立评估验证集

按最高验证AP选择同一checkpoint，不能拼接不同轮的指标峰值。

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为已完成point_strip目录
CKPT=$(python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
import yaml

run = Path(sys.argv[1])
cfg = yaml.safe_load((run / 'resolved_config.yml').read_text())
rows = [json.loads(line) for line in (run / 'log.txt').read_text().splitlines() if line.strip()]
by_epoch = {r['epoch']: r for r in rows if 'test_coco_eval_bbox' in r}
best = max(by_epoch.values(), key=lambda r: r['test_coco_eval_bbox'][0])
stop = cfg['train_dataloader']['collate_fn']['stop_epoch']
print(run / ('best_stg2.pth' if best['epoch'] >= stop else 'best_stg1.pth'))
print('val_epoch=', best['epoch'], 'AP=', best['test_coco_eval_bbox'][0] * 100, file=sys.stderr)
PY
)
EVAL=$(mktemp -d "$OUT/val_eval_XXXXXX")
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
  torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$CKPT" --test-only --seed 0 --use-amp \
  --output-dir "$EVAL" --summary-dir "$EVAL/summary" \
  > "$EVAL/console.log" 2>&1 < /dev/null &
EVAL_PID=$!
printf '%s\n' "$EVAL_PID" > "$EVAL/launcher.pid"
disown "$EVAL_PID"
printf 'EVAL=%s PID=%s CHECKPOINT=%s\n' "$EVAL" "$EVAL_PID" "$CKPT"
tail -f "$EVAL/console.log"
```

本命令使用原验证集，独立结果在$EVAL/eval.pth及console.log，不覆盖训练输出。
若评估现有测试集，在torchrun参数中追加
`-u val_dataloader.dataset.ann_file=/home/zxy/sar/datasets/SAR-AIRcraft-1.0/COCO/test.json`。

## 本地功能验证与范围

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_rem_point_strip.py -v
```

验证针对点旁路与轴向空间支持、保留权重的初始化、参数数量、单次DWT/IDWT、奇数尺寸、
FP32/BF16梯度、实际四层Decoder与原MAL/FGL/DDF和DN/aux损失、EMA/保存恢复以及完整协议一致性。
另运行原REM、空间相位/方形控制与per_band相关验证，检查既有路径兼容性。
本次共33项通过：新增8项、既有相关25项。启动器、其中3段Python代码和本文5段Bash命令语法验证通过。
本地CPU验证不等于服务器CUDA FP16/SyncBN双卡训练；当前没有该变体的训练结果。
