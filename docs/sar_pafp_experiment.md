# SAR-AIRcraft PAFP：58 轮单模块实验

本次选择 PAFP 先试：以可学习的空间 / 通道先验、动态阈值和多尺度高斯混合，
适应 SAR 场景中变化的细节幅度与背景尺度。这是待训练验证的选择，不是已有增益结论。
原文 [LFP 第3.1节](https://arxiv.org/html/2508.06878)以低频空间引导、固定幅值门限和可学习 sigma 的高斯滤波净化高频。
本实现参照该主线并结合用户提供的 PAFP 改造描述独立编写，不宣称复现未知第三方 PAFP 源码；
不引入原 NS-FPN 的 SFS 模块或额外 CUDA 编译依赖。

实现：`engine/deim/prior_adaptive_frequency.py`；接在 HybridEncoder stride-8、256 通道投影特征上，FPN 融合之前。
配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pafp_58e_b28_old_server.yml`。

## 实现选择

- 1×1 投影 / GroupNorm / SiLU 压到64通道，做一级有符号 Haar 分解；支持奇数尺寸的填充 / 裁剪。
- LL 的通道均值 / 最大值生成空间先验；LL 全局统计预测三个高频子带的通道先验。
- 两个先验调制高频后，使用每样本 / 每子带 RMS 归一化幅值；LL 预测0.1..1.0的相对阈值。
  `sigmoid(10 * (normalized_abs - threshold))` 平滑保留强响应，对较弱响应采用高斯混合。
- 三个高斯专家为3×3、5×5、7×7，sigma 初始0.6、1.0、1.4，通过 softplus+0.25 保持正值并学习。
  LL 预测 softmax 专家权重；滤波逐通道处理有符号系数，使用 replicate padding，幅值只用于门控。
- LL 原样参与 IDWT；重建差分和深度卷积 / 点卷积细化构成校正量，再投影回256通道。
  输出为 `X + alpha * delta`，仅 alpha 初始为零，保留原预训练特征路径。
- 适配器在 AMP 中用 FP32 计算，再返回输入 dtype；不引入新标签、辅助 loss 或检测头改动。

只增加 PAFP，不叠加 REM、PACFG、CrossGate-DBDM 或其他候选。
训练日志增加 `train_pafp_alpha`，元数据保存 PAFP 开关与瓶颈宽度。
完整继承旧服务器58轮 baseline：DEIM-D-FINE-M / HGNetv2-B2，七类0–6，train3047 / val442 / test879，
seed0，GPU0/1，全局 batch28，基础 / 验证尺寸640，58轮，warmup432，flat29、no_aug8、lr_gamma0.5；
增强 policy [4,29,50]、mixup [4,29]、stop50，AdamW 主LR3.5e-4、骨干非BN LR3.5e-5、WD1e-4。
原多尺度 / Mosaic / MixUp、EMA、AMP、FP32 loss / retry 和评估口径保持一致。
使用 COCO M 权重 `-t` 初始化；启动器保存配置、Git revision、seed、全部参数、权重路径和启动命令。
已有数据无需再次准备。

## 旧服务器更新与启动

Git 有未提交改动或分叉时先查看并保留现有文件，不要强制覆盖。

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
  configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_pafp_58e_b28_old_server.yml
```

启动器建立新实验目录，使用 `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2`，
以 `nohup ... > console.log 2>&1 < /dev/null &` 启动，先保存 `$!` 到 `launcher.pid`，再 `disown`。
当前开发环境无法直连旧服务器；以上命令需要在服务器执行。本地没有远程启动或训练完成证据。

## 重连、日志、指标和恢复

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为实际PAFP训练目录
tail -f "$OUT/console.log"
```

同一终端 Ctrl+C 退出日志跟踪后，分别运行：

```bash
nvidia-smi -i 0,1
ps -p "$(cat "$OUT/launcher.pid")" -o pid,ppid,etime,args
tail -n 3 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

逐轮指标在 `log.txt`，TensorBoard 在 `summary/`，训练 COCO 评估数据在 `eval/latest.pth`。
`test_coco_eval_bbox` 第0、1、2项为 val AP / AP50 / AP75，报告时乘100。

只在训练进程已停止时恢复：

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原PAFP训练目录
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

恢复使用解析配置和 `-r last.pth`，不同时提供 `-t`；跟踪打印的 `resume_*.console.log`。
当前 `last.pth` 正常最后更新于 epoch49，不是最终权重。

## 按 val AP 选择模型，再做 test 评估

```bash
cd /home/zxy/sar/repos/deim
conda activate deim_zxy
export OUT=/home/zxy/sar/experiments/替换为原PAFP训练目录

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

按 val AP 选择 `best_stg1` / `best_stg2`，不按阶段名称或 test AP 重新选择。
独立评估数据在 `$EVAL/eval.pth`，控制台输出 test AP / AP50 / AP75 / AR100。

## 本地定向验证

```powershell
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_prior_adaptive_frequency.py -v
conda run --no-capture-output -n pytorch python -m unittest discover -s tests -p test_sar_module_integration.py -v
```

验证针对高斯滤波的归一化 / 保号 / sigma 梯度、强弱高频响应、动态分支梯度、CPU autocast、奇数尺寸、
Encoder 实际路由和配置继承；不使用真实图片 / 预训练权重，不启动完整训练。
结果：PAFP 定向验证6项、现有模块回归8项全部通过；Python / Bash 和交接命令语法检查通过。
CUDA FP16 / 双卡及实际检测收益由服务器实验确认。
