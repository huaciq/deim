# SAR-AIRcraft REM：频带基与方向条带机制消融

## 实验定义

原REM的验证集AP约64.2，高于58轮baseline约63.1；REM+W-FDR未超过REM。
本次新增两项独立机制对照，各自从同一COCO初始化新训，与已报告的原REM比较。

| 组别 | 2x2分析/合成 | 中间两路卷积 | 用途 |
|---|---|---|---|
| baseline（已有） | 无适配器 | 无适配器 | 原检测器 |
| REM（已有） | Haar / IDWT | 原方向条带 | 完整方法 |
| REM-space-to-depth（新增） | 四个原始像素相位 / 重排还原 | 原方向条带 | 检验Haar频带基相对空间分块的贡献 |
| REM-square（新增） | Haar / IDWT | 方形3x3和1x1 | 检验显式横纵条带的贡献 |

两项均接在stride-8 input_proj之后、FPN/PAN融合之前；输入输出仍为256通道、相同空间大小。
原REM不指定rem_variant时仍使用haar_directional，其计算和checkpoint键保持原样。
新配置仅增加HybridEncoder.rem_variant及独立输出/文档路径，不开启W-FDR或其他方法。

### A：space_to_depth，保留完全相同的卷积容量

用PixelUnshuffle(2)将每通道2x2块拆成[左上、右上、左下、右下]四个原始空间相位，
逐相位送入原来的四路卷积，再用PixelShuffle(2)重排还原。
相位与原分支ll/lh/hl/hh依次对应；这些名称仅保留代码兼容性，不再表示频带。
不做求和差、滤波或额外幅值缩放，不以零张量替代高频。
奇数尺寸与原Haar一样在右/下replicate padding，重建后裁回原尺寸。
四路卷积的权重初始化、通道、核、BN/激活、alpha及分支数与原REM完全一致。

它控制了四路空间处理、下采样尺度和卷积容量。与原REM的差别是分析/合成基，
不是普通原分辨率堆卷积，因此结论应限定为Haar混合基相对原始空间相位的价值。
即使Haar胜出，也不能仅凭此证明雷达散射物理解释。

### B：haar_square，移除显式方向条带设计

保留Haar/IDWT、LL和HH原分支、LH/HL入口1x1与末尾3x3。
只将LH/HL的条带卷积对替换为：

```text
原 LH: 1x1 C->C, 1x3 C->C, 3x1 C->C, 3x3 C->C
原 HL: 1x1 C->C, 3x1 C->C, 1x3 C->C, 3x3 C->C
新两路: 1x1 C->C, 3x3 C->M, 1x1 M->C, 3x3 C->C
M = max(1, round(3*C/5)); C=256时M=154
```

每个Conv后仍为BN和原SiLU；每条路径仍4层卷积、有效感受野5x5。
原条带对含6*C*C个卷积权重，替换对含10*C*M个，故以M约0.6C匹配容量。
主干和模块输入输出仍为256通道；仅这两对卷积使用154通道中间层。
先构造原REM，再隔离随机状态替换指定卷积，保留共同分支参数和后续RNG。

方形卷积也可以学到方向模式，Haar的方向子带也仍存在；此对照不等于消除全部方向信息。
它还改变了中间通道分配及一处BN宽度，属于近似容量匹配的结构对照，不能声称完全无混杂。

| 适配器 | 可训练参数 | 640输入时适配器卷积GMACs |
|---|---:|---:|
| 原REM | 2,823,681 | 4.508877 |
| space_to_depth | 2,823,681 | 4.508877 |
| haar_square | 2,825,321 | 4.512154 |

square参数差+1,640（约0.058%），卷积乘加差约+0.073%。
MAC仅计80x80的stride-8特征进入适配器后的卷积，不含变换、BN、激活和残差，不是GPU时延。
所有组均为Y=X+alpha*delta、alpha零初始化；非残差权重正常初始化。

## 文件与固定协议

- engine/deim/radar_enhancement.py：可逆空间相位变换和方形卷积模式。
- engine/deim/hybrid_encoder.py：传递rem_variant，原插入位置和默认值保留。
- tools/run_sar_module_experiment.sh：记录variant、变换、相位顺序和方形中间通道。
- tests/test_rem_mechanism_ablations.py：解析变换、容量/感受野、梯度及真实检测路径验证。
- configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_space_to_depth_58e_b28_old_server.yml。
- configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_square_58e_b28_old_server.yml。

两配置均单父级继承原REM配置：DEIM-D-FINE-M / HGNetv2-B2、7类ID0–6、train3047/val442/test879；
seed0、58轮、全局batch28（双卡每卡14）、基础及验证640、训练多尺度480–800；
warmup432、flat29、no_aug8、lr_gamma0.5；增强policy [4,29,50]、mixup [4,29]、stop50；
AdamW主LR3.5e-4、骨干非BN LR3.5e-5、WD1e-4，EMA、SyncBN、AMP FP16、FP32 loss/retry。
RandomZoomOut和RandomIoUCrop仍全程关闭，其余增强沿用原协议。
Decoder点数仍[3,6,3]、原分类/LQE/FDR/FGL/DDF及top300后处理保持原协议，无新增损失。
数据和0–6标注沿用现有SAR-AIRcraft目录，无需准备新数据。
每组均从COCO M权重-t初始化、使用新优化器；不继续训练原REM checkpoint。
启动器归档配置、Git revision、seed、batch、输入、优化器、增强、EMA/AMP、初始化和评估协议。

## 服务器更新，先跑空间分块对照

若服务器工作树有改动或分叉，先检查并保留，不使用强制覆盖。然后执行：

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_space_to_depth_58e_b28_old_server.yml
```

启动器新建实验目录，使用CUDA_VISIBLE_DEVICES=0,1、torchrun --standalone --nproc_per_node=2 train.py；
以nohup后台执行，console.log接收stdout/stderr，stdin来自/dev/null，保存$!至launcher.pid再disown。
两卡不空闲时原启动器不启动、不排队、不停止已有运行。

第一组结束、两卡空闲后，再单独启动方形卷积对照：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export EXPERIMENT_ROOT=/data2/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth
bash tools/run_sar_module_experiment.sh \
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_rem_square_58e_b28_old_server.yml
```

## 日志、进程、指标与TensorBoard

重连后激活环境并将OUT设为启动器打印的对应消融目录：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为实际REM消融目录
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tail -f "$OUT/console.log"
```

Ctrl+C退出跟踪后，读取最佳AP检查点整组指标和残差alpha：

```bash
python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys
import yaml

run = Path(sys.argv[1])
cfg = yaml.safe_load((run / 'resolved_config.yml').read_text())
rows = [json.loads(s) for s in (run / 'log.txt').read_text().splitlines() if s.strip()]
by_epoch = {r['epoch']: r for r in rows if 'test_coco_eval_bbox' in r}
rows = [by_epoch[e] for e in sorted(by_epoch)]
best = max(rows, key=lambda r: r['test_coco_eval_bbox'][0])
print('rem_variant=', cfg['HybridEncoder'].get('rem_variant', 'haar_directional'))
for label, row in [('best_AP_checkpoint', best), ('last_logged', rows[-1])]:
    print(label, 'epoch=', row['epoch'], 'AP/AP50/AP75/AP_S/AP_M/AP_L=',
          [round(v * 100, 4) for v in row['test_coco_eval_bbox'][:6]],
          'rem_alpha=', row.get('train_rem_alpha'))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

log.txt为JSONL指标；COCO bbox第0/1/2项为AP/AP50/AP75，百分数报告乘100。
TensorBoard在summary/，训练评估数据在eval/latest.pth。
各组按最佳val AP选同一checkpoint报告所有指标，不拼接各指标独立峰值。
原REM若胜出，分别说明该结构中Haar基或条带设计有候选贡献；单seed不宣称显著性或物理机理已被证明。

## 恢复对应消融运行

原进程停止后使用其保存配置及-r last.pth，不同时传-t：

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为原REM消融目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复日志为打印的resume_*.console.log；last.pth正常最后更新于epoch49，不是最终第57轮。
保存配置的rem_variant决定恢复哪一种变换和卷积结构。

## 按验证AP选权重并评估测试集

```bash
cd /home/zxy/sar/repos/deim
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export OUT=/data2/zxy/sar/experiments/替换为原REM消融目录
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

按实际验证AP选best_stg1.pth/best_stg2.pth；独立评估数据在$EVAL/eval.pth，指标见控制台。

## 本地功能验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_rem_mechanism_ablations.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_radar_enhancement.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_pgd_rem.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sar_58e_protocol.py -v
```

验证相位顺序、奇偶尺寸重建和梯度、精确分支路由、参数/感受野、共同初始化/RNG、
零残差两步学习、CPU BF16 AMP、stride-8插入、4层普通/aux/DN原损失训练、EMA/checkpoint及配置协议。
本地未执行真实SAR数据训练或CUDA双卡吞吐/显存测量，服务器启动以STARTED行、PID和日志为准。

验证结果：新增10项、原REM 5项、PGD-REM 6项、58轮协议5项，共26项全部通过。
通过真实配置工厂核对全网络可训练参数：原REM及空间分块组均22,301,250，方形卷积组22,302,890。
对256通道80x80输入逐卷积统计的MAC与上表一致；启动器、本文6段Bash及5段嵌入Python语法检查通过。
