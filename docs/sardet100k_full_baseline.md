# SARDet-100K 全集 baseline：固定 120e/b32 协议

配置：`configs/deim_dfine/deim_hgnetv2_s_sardet100k_baseline_120e_b32_paper_nozoom_nocrop.yml`。
直接继承已固定的 [子集协议](sardet_paper120_nozoom_nocrop.md)，仅替换 train/val 标注和
默认输出目录。原子集配置、结果和诊断产物保留。

## 固定条件

| 项目 | 设置 |
| --- | --- |
| 训练数据 | 官方 Annotations/train.json，94,493 张、198,747 个目标 |
| 验证数据 | 官方 Annotations/val.json，10,492 张、22,703 个目标，每轮完整验证 |
| 图像根目录 | /root/zxy/TCRA-SAR/datasets/SARDet_100K/JPEGImages/train 与 val |
| 类别 | 0..5，ship/aircraft/car/tank/bridge/harbor，remap_mscoco_category=False |
| 初始化 / seed | COCO 检测器权重 -t，fresh optimizer，seed=0，不从子集检查点恢复 |
| 训练资源 | GPU 0、1；global batch=32，每卡 16；120 轮 |
| 验证资源 | global batch=64，每卡 32；640 输入 |
| 输入 / 多尺度 | 基础 640；480..800、步长 32、640 权重 20/30；epoch 118 起固定 640 |
| AdamW | 主 LR=4e-4、骨干 LR=2e-4、betas=(0.9,0.999)、WD=1e-4，norm/BN WD=0 |
| 学习率 | 二次预热 2000 步，flat 到 epoch 64，再余弦至 epoch 118，最小 LR 为初始值 0.5 倍 |
| 增强 | Mosaic/MixUp 各 p=0.5，4≤epoch<64；PhotometricDistort p=0.5，4≤epoch<118 |
| 关闭增强 | RandomZoomOut、RandomIoUCrop 全程关闭；水平翻转 p=0.5 保留 |
| EMA / 数值策略 | EMA=0.9999、warmups=1000；FP16、GradScaler、FP32 loss、非有限批次一次 FP32 重算 |
| 模型 / 评估 | 原 DEIM-D-FINE-S、SiLU、BN 不冻结、SyncBN、clip norm=0.1；六类 COCO bbox，无平移或分数校准 |
| 进程资源 | ulimit -Sn 65535，继承至两个 rank 和 DataLoader workers |

沿用现有模型代码中的 FDR 诊断输出附加键；预测 logits/boxes 和参数没有改动，
此前定向回归已验证原输出逐元素一致。没有加入 FDR 方差模块或新损失。

两卡 DistributedSampler + 每卡 16/drop_last 下，训练 **2952 steps/epoch**，
合计 **354240 steps**；warmup 仍为 2000 步（约 0.678 个全集 epoch），不因数据变大缩放。
flat_iter=188928、no_aug_iter=5904、收尾起点 step=348336；每卡验证 164 批。
这些是按全集图数与固定 batch 推导，启动日志中的实际 scheduler 参数为最终依据。

## 更新和从 COCO 启动

官方全集标注已存在，标签为 0..5，file_name 是裸文件名，无需数据转换或重新生成 JSON。
先查看 Git 状态；如有已跟踪修改或分叉先查看内容，保留无关未跟踪文件，不强制覆盖。

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
CONFIG=configs/deim_dfine/deim_hgnetv2_s_sardet100k_baseline_120e_b32_paper_nozoom_nocrop.yml
OUT=$(mktemp -d /root/zxy/experiments/deim_s_sardet100k_paper120_b32_nozoom_nocrop_fd65535_seed0_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_XXXXXX)
export CONFIG OUT
git rev-parse HEAD > "$OUT/git_revision.txt"
cp "$CONFIG" "$OUT/source_config.yml"
printf '0\n' > "$OUT/seed.txt"
python - <<'PY'
import collections, json, math, os, pathlib, yaml
from engine.core.yaml_utils import load_config
p=pathlib.Path(os.environ['OUT'])
cfg=load_config(os.environ['CONFIG'], {})
cfg.pop('__include__', None)
cfg.update(output_dir=str(p), summary_dir=str(p/'summary'), seed=0, use_amp=True)
(p/'resolved_config.yml').write_text(yaml.safe_dump(cfg, sort_keys=False))
(p/'config_path.txt').write_text(str(p/'resolved_config.yml')+'\n')
splits={}
for name in ['train','val']:
    loader=cfg[name+'_dataloader']
    d=json.loads(pathlib.Path(loader['dataset']['ann_file']).read_text())
    splits[name]=dict(annotation=loader['dataset']['ann_file'],
        image_folder=loader['dataset']['img_folder'],images=len(d['images']),
        objects=len(d['annotations']),categories=d['categories'],
        category_counts=dict(sorted(collections.Counter(a['category_id'] for a in d['annotations']).items())))
steps=math.ceil(splits['train']['images']/2)//(cfg['train_dataloader']['total_batch_size']//2)
meta=dict(source_config=os.environ['CONFIG'],git_revision=(p/'git_revision.txt').read_text().strip(),
    seed=0,devices=[0,1],world_size=2,global_batch_size=32,per_gpu_batch_size=16,
    initialization='/root/zxy/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth',
    initialization_mode='COCO tuning (-t), fresh optimizer, no resume',
    file_descriptor_soft_limit=65535,steps_per_epoch=steps,total_steps=steps*cfg['epoches'],
    splits=splits,resolved_config=cfg,
    numerical_exception_policy='FP16 + FP32 loss; nonfinite batch replayed once in FP32',
    protocol='Fixed 120e/b32 protocol; full official train and val; only dataset annotation paths changed')
(p/'dataset_stats.json').write_text(json.dumps(splits,indent=2))
(p/'run_metadata.json').write_text(json.dumps(meta,indent=2))
PY
cat > "$OUT/start_command.sh" <<EOF
#!/usr/bin/env bash
set -e
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
ulimit -Sn 65535
ulimit -Sn > "$OUT/nofile_soft.txt"
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

服务器 GitHub HTTPS 超时时，通过 Git bundle 传输提交，服务器执行
`git fetch /tmp/本次.bundle main` 和 `git merge --ff-only FETCH_HEAD`，不用直接覆盖源码。
实际运行路径、PID 和提交另记录在交付 RUN.md 及实验目录中。

## 重连、日志、GPU、指标及 TensorBoard

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换为本次实际目录
export OUT
tail -f "$OUT/console.log"
```

其它终端分别执行：

```bash
OUT=/root/zxy/experiments/替换为本次实际目录
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
ps --ppid "$(cat "$OUT/launcher.pid")" -o pid,etime,args
cat "/proc/$(cat "$OUT/launcher.pid")/limits" | grep 'Max open files'
watch -n 2 nvidia-smi
grep 'AMP_FP32_RETRY' "$OUT/console.log"
```

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换为本次实际目录
export OUT
python - <<'PY'
import json,os,pathlib
p=pathlib.Path(os.environ['OUT'])/'log.txt'
if not p.exists():
    print('首次完整验证尚未写入 epoch 指标')
else:
    rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    for name,r in [('best',max(rows,key=lambda r:r['test_coco_eval_bbox'][0])),('latest',rows[-1])]:
        print(name,'epoch',r['epoch'],dict(zip(['AP','AP50','AP75','APs','APm','APl'],
              [round(100*x,3) for x in r['test_coco_eval_bbox'][:6]])))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

原始 COCO 指标 0..1，上述命令乘 100。首轮训练和完整验证后 log.txt 才有 AP。
TensorBoard 写 summary/，最近评估写 eval/latest.pth。比较 best_stg1/best_stg2 的实际 AP
后选择最佳权重，不凭阶段文件名选择。

## 中断后的恢复

只对已经停止的本次训练执行，不与原进程并行；-r 和 -t 不同时使用。
当前 solver 的 last.pth 只在 epoch<118 更新，正常最终更新到 epoch 117，不是最终权重。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换为本次实际目录
LOG="$OUT/resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).console.log"
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/last.pth" \
  --seed 0 --use-amp --output-dir "$OUT" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
tail -f "$LOG"
```

## 全集验证检查点

训练期间已经每轮使用完整 10,492 张验证图。若需重新评估，下面以 stage1 为例，
stage2 存在时改 STAGE=best_stg2；两者均以实测 AP 比较，每次使用新的输出目录。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/替换为本次实际目录
STAGE=best_stg1
EVAL=$(mktemp -d "$OUT/fullval_${STAGE}_XXXXXX")
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/$STAGE.pth" --test-only --use-amp \
  --output-dir "$EVAL" --summary-dir "$EVAL/summary" \
  > "$EVAL/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$EVAL/launcher.pid"
disown "$PID"
tail -f "$EVAL/console.log"
```
