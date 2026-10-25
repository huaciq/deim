# SARDet 固定 baseline：120 轮、batch 32、关闭 ZoomOut/IoUCrop

配置：`configs/deim_dfine/deim_hgnetv2_s_sardet10pct_baseline_120e_b32_paper_nozoom_nocrop.yml`。
直接继承官方 DEIM-D-FINE-S 配置，不继承此前自定的 40/80/120 轮调度。
后续模块实验固定使用本协议，原配置、日志和权重保留。

## 论文与公开代码的差别及本次取值

[论文 §4.1](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.pdf)
说明前 4 轮增强预热、后半程关闭 Dense O2O、最后 2 轮关闭增强。
[补充材料表 9、图 5](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Huang_DEIM_DETR_with_CVPR_2025_supplemental.pdf)
给出 S 的 120 轮及学习率，调度图是一个 60 轮的示例。
[公开 S 配置](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/deim_dfine/deim_hgnetv2_s_coco.yml)
实际为 132 轮，注释说明为了探索 EMA 延长训练，flat=64，no_aug=12，增强边界 [4,64,120]。
[公开 DEIM 基础配置](https://github.com/Intellindust-AI-Lab/DEIM/blob/main/configs/base/deim.yml)
给出 2000 步预热及其余增强参数。

用户明确要求总计 120 轮并遵循论文。因此本次固定总轮数 120、保留 S 的
flat/Dense 边界 64，按论文正文将最后两轮设为 epoch 118、119。
这是依据论文正文和公开 S 配置明确补全的 120 轮协议，不声称论文提供了完全相同的
120 轮 YAML，也不沿用此前自定的 warmup=1000、flat=60、stop=112。

| 项目 | 固定值 |
| --- | --- |
| 数据 | 固定 SARDet 六类筛选子集，train 9,449 / val 4,096；不重新抽样 |
| 初始化 / seed | COCO 完整检测器权重，`-t`；seed 0 |
| GPU / 训练 batch | GPU 0、1；全局 32，每卡 16 |
| 总轮数 / 输入 | 120；基础和验证 640；训练保留官方多尺度 |
| AdamW | 主 LR=0.0004，backbone LR=0.0002；betas=(0.9,0.999)，WD=0.0001；norm/BN WD=0 |
| 学习率 | 二次预热 2000 steps；至 epoch 64 平坦；余弦至 epoch 118；最后两轮维持 0.5 倍 LR |
| 最小学习率 | 主 LR=0.0002，backbone LR=0.0001 |
| Mosaic / MixUp | 均 p=0.5，epoch 4≤e<64；MixUp 混合比例 Uniform(0.45,0.55) |
| Mosaic 内部参数 | output=320；旋转 ±10°、平移 0.1/0.1、缩放 0.5..1.5；缓存关闭 |
| 其它增强 | PhotometricDistort p=0.5，4≤e<118；水平翻转 p=0.5 贯穿训练 |
| 多尺度 | epoch<118 使用 480..800，步长 32，640 权重 20/30；之后固定 640 |
| 关闭的增强 | RandomZoomOut、RandomIoUCrop 从 ops 列表中移除 |
| EMA / AMP | EMA decay=0.9999、warmups=1000；FP16、GradScaler；clip norm=0.1 |
| 模型 / 损失 | 原 S、SiLU、BN 不冻结、SyncBN；MAL gamma=1.5 及官方损失权重不变 |
| 验证 | 全局 batch=64，每卡 32；逐轮 COCO bbox AP/AP50/AP75/APs 等，无平移校准 |

轮号均为从零开始的日志 epoch。epoch 118 起停止 PhotometricDistort、多尺度；Mosaic、
MixUp 已在 64 停止。普通水平翻转仍按官方实现保留，因此“关闭增强”准确说是关闭强增强。
9449 张、全局 batch 32、两卡 drop_last 后每轮 295 steps，总计 35,400 steps。
2000 步预热约占 6.78 个子集 epoch；按用户要求保持官方步数，不按子集大小缩短。

数据类别/路径、COCO 迁移初始化是 SAR 任务适配。保留已有 FP32 loss 及异常批次一次
FP32 重算机制，以处理此前 FP16 非有限值；这是显式记录的工程差异，不是论文提出的方法。
这次不修改训练引擎。子集结果只用于方向筛选，方法效果仍需在全集独立验证。

## Git 更新及从 COCO 重新启动

服务器已有子集，无需再次准备。先查看 Git 状态；存在已跟踪修改或分叉时先查看具体内容，
保留无关未跟踪文件，不使用 reset 或强制 checkout。

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
CONFIG=configs/deim_dfine/deim_hgnetv2_s_sardet10pct_baseline_120e_b32_paper_nozoom_nocrop.yml
OUT=$(mktemp -d /root/zxy/experiments/deim_s_sardet10pct_paper120_b32_nozoom_nocrop_seed0_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_XXXXXX)
export CONFIG OUT
git rev-parse HEAD > "$OUT/git_revision.txt"
cp "$CONFIG" "$OUT/source_config.yml"
cp /root/zxy/experiments/data_subsets/sardet100k_screen_v1/stats.json "$OUT/subset_stats.json"
printf '0\n' > "$OUT/seed.txt"
python - <<'PY'
import json, os, pathlib, yaml
from engine.core.yaml_utils import load_config
p = pathlib.Path(os.environ['OUT'])
cfg = load_config(os.environ['CONFIG'], {})
cfg.pop('__include__', None)
cfg.update(output_dir=str(p), summary_dir=str(p/'summary'), seed=0, use_amp=True)
(p/'resolved_config.yml').write_text(yaml.safe_dump(cfg, sort_keys=False))
(p/'config_path.txt').write_text(str(p/'resolved_config.yml')+'\n')
meta = dict(source_config=os.environ['CONFIG'], git_revision=(p/'git_revision.txt').read_text().strip(),
    seed=0, devices=[0,1], world_size=2, global_batch_size=32, per_gpu_batch_size=16,
    initialization='/root/zxy/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth',
    initialization_mode='COCO tuning (-t), fresh optimizer, no resume',
    protocol_note='120 total; official S flat/Dense=64; paper last-two-epoch tail=118; disable ZoomOut/IoUCrop',
    numerical_exception_policy='FP16 + FP32 loss; same nonfinite batch replayed once in FP32',
    resolved_config=cfg)
(p/'run_metadata.json').write_text(json.dumps(meta, indent=2))
PY
cat > "$OUT/start_command.sh" <<EOF
#!/usr/bin/env bash
set -e
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
exec torchrun --standalone --nproc_per_node=2 train.py \\
  -c "$OUT/resolved_config.yml" \\
  -t /root/zxy/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \\
  --seed 0 --use-amp --output-dir "$OUT"
EOF
nohup bash "$OUT/start_command.sh" > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
printf 'OUT=%s\nPID=%s\n' "$OUT" "$PID"
```

服务器 GitHub HTTPS 超时时，从开发端创建包含本次提交的 Git bundle 并通过 SSH 传输，
服务器执行 `git fetch /tmp/本次.bundle main`、`git merge --ff-only FETCH_HEAD` 完成同一 Git 更新。
本次实际运行路径、提交、PID 另记录在交付的 RUN.md 和实验目录中。

## 重连、控制台、GPU、指标、TensorBoard

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换为本次实际目录
export OUT
tail -f "$OUT/console.log"
```

以下分别在其它终端运行：

```bash
OUT=/root/zxy/experiments/替换为本次实际目录
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
watch -n 2 nvidia-smi
grep 'AMP_FP32_RETRY' "$OUT/console.log"
```

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换为本次实际目录
export OUT
python - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['OUT'])/'log.txt'
rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
for name, r in [('best', max(rows, key=lambda r:r['test_coco_eval_bbox'][0])), ('latest', rows[-1])]:
    print(name, 'epoch', r['epoch'], dict(zip(['AP','AP50','AP75','APs','APm','APl'],
          [round(100*x,3) for x in r['test_coco_eval_bbox'][:6]])))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

首次验证完成后 log.txt 才有 AP；原指标 0..1，以上乘 100。
TensorBoard 写入 summary/，最近一次 COCO 评估写入 eval/latest.pth。

## 今后中断时恢复（本次执行的是从 COCO 重新开始）

确认原任务已退出再执行。当前 solver 的 last.pth 仅在 epoch<118 更新，正常运行最终
更新到 epoch 117，不能称为最终权重。最后阶段中断可能需要从该点重复部分轮次。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换为本次实际目录
LOG="$OUT/resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).console.log"
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/last.pth" \
  --seed 0 --use-amp --output-dir "$OUT" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
tail -f "$LOG"
```

## 检查点评估

分别评估实际存在的 best_stg1.pth / best_stg2.pth，以实测 AP 选择；不凭阶段文件名选最优。
以下使用训练时相同的 4,096 张筛选验证图。将 STAGE 改为 best_stg2 可评估另一权重。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换为本次实际目录
STAGE=best_stg1
EVAL=$(mktemp -d "$OUT/subsetval_${STAGE}_XXXXXX")
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/$STAGE.pth" --test-only --use-amp \
  --output-dir "$EVAL" --summary-dir "$EVAL/summary" \
  > "$EVAL/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$EVAL/launcher.pid"
disown "$PID"
tail -f "$EVAL/console.log"
```

如需完整 10,492 张验证图，在上述命令中添加
`-u val_dataloader.dataset.ann_file=/root/zxy/TCRA-SAR/datasets/SARDet_100K/Annotations/val.json`，
并使用新的 fullval 评估目录；完整验证与筛选验证结果分别报告。
