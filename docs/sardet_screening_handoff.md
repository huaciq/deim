# SARDet-100K 六类方向筛选子集

本方案用于前期选择方法方向。固定训练/验证图像清单，让每个候选与相同条件的
DEIM-D-FINE-S 基线比较。此次只准备数据和配置，不启动训练。

| 阶段 | 训练图像 | 验证图像 | 轮数 | 训练种子 | 用途 |
|---|---:|---:|---:|---|---|
| 快速筛选 | 9,449 | 4,096 | 40 | 0 | 观察明显收益、代价和收敛趋势 |
| 入围确认 | 同一批 9,449 | 官方完整 10,492 | 80 | 0、1 | 基线和入围方法成对复跑 |
| 正式实验 | 官方完整 94,493 | 官方完整 10,492 | 单独确定正式协议 | 多种子 | 确认全量收益，不能用子集成绩替代 |

80 轮配置从 COCO 权重重新初始化，不从 40 轮检查点延长训练。40/80 轮结果只与
各自同协议基线比较。测试集不参与筛选、选权重或调参。

## 数据组成和抽样

复用服务器已有 `Annotations/train_10pct_seed42.json` 的 9,449 个训练图像 ID，
从官方 `train.json` 提取每张图的全部原始标注，输出与已有训练子集一致。
这份子集按类别抽样，包含 20,661 个目标。继续缩到一两千张会使少数类场景过少。

验证集只从官方 `val.json` 选择，固定 seed=42。按“图内类别组合 × 图内目标面积中位数
（缩放到 640 后）× 目标密度 × 图像分辨率”划为 106 层，以原有比例分配 4,096 个
名额，然后在层内随机选图。每层配额与理想比例之差小于一张图。按图选取，保留
该图全部标注，不截掉其它类别，也不修改 bbox、area、category_id 或 file_name。

| 类别（ID） | 训练目标数 | 筛选验证目标数 | 训练含该类图像数 | 筛选验证含该类图像数 |
|---|---:|---:|---:|---:|
| ship (0) | 9,690 | 4,072 | 6,187 | 2,512 |
| aircraft (1) | 4,279 | 2,041 | 1,497 | 772 |
| car (2) | 1,009 | 488 | 84 | 41 |
| tank (3) | 2,597 | 880 | 208 | 60 |
| bridge (4) | 2,781 | 1,266 | 1,526 | 680 |
| harbor (5) | 305 | 158 | 134 | 65 |
| 合计 | 20,661 | 8,905 | 9,449 张唯一图像 | 4,096 张唯一图像 |

训练子集在 640 输入下的小目标占比为 26.3%（全量训练为 23.9%），筛选验证为 25.6%
（完整验证为 25.8%）。训练子集略偏小目标，适用于这里的小目标方向筛选；它不是
完整数据分布的精确替身。标准 COCO APs 使用原始标注面积，不能与“640 输入下
面积小于 32²”的抽样统计混为一谈；stats.json 分别记录两种统计。

原 JSON 使用匿名数字文件名，没有来源数据集字段，因此此方案没有宣称按原始
来源数据集或大场景分层。[官方数据说明](https://github.com/zcablii/SARDet_100K)
列出的 SARDet-100K 来源包括 OGSOD，所以它不能当作与 OGSOD 完全独立的外部数据集。
本次不尝试恢复未知的原始场景分组，沿用官方划分。

子集目录固定为：

```text
/root/zxy/experiments/data_subsets/sardet100k_screen_v1/
  train.json
  val.json
  image_ids.json
  stats.json
```

图像继续从 `SARDet_100K/JPEGImages/train`、`JPEGImages/val` 读取，不复制图像。
该数据集的 file_name 没有 split 前缀；这与 OGSOD 的根目录用法不同。
所有方法和训练种子共用这份清单，不能每次重新抽样。生成脚本要求新输出目录，
后续已生成则直接使用，重新设计抽样时另建版本。

## 固定训练协议

- 配置：`configs/deim_dfine/deim_hgnetv2_s_sardet10pct_screen_40e.yml`；
  入围确认：`configs/deim_dfine/deim_hgnetv2_s_sardet10pct_confirm_80e.yml`。
- 模型：DEIM-D-FINE-S，HGNetv2-B0，六类，类别 ID 0..5，关闭 COCO 类别重映射。
- 初始化：`/root/zxy/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth`，使用 `-t`；
  不使用此前在 OGSOD 上训练过的权重来初始化筛选基线。
- 两张 4090，GLOBAL batch=28（每卡 14），640 为基础/验证输入；训练保留已有多尺度。
- AdamW，主学习率 0.0004、骨干 0.0002，weight_decay=0.0001，lr_gamma=0.5。
- 40 轮：warmup=500 iterations，flat=20 epochs，最后 4 轮进入现有无强增强阶段；
  80 轮：warmup=500，flat=40，最后 8 轮进入对应阶段。
- 保留 Mosaic、PhotometricDistort、水平翻转、MixUp 和现有多尺度；移除 ZoomOut、
  RandomIoUCrop。40 轮 policy=[2,20,36]，MixUp=[2,20]，多尺度至 epoch 36；
  80 轮对应 [4,40,72]、[4,40]、epoch 72。所谓 no-aug 阶段仍保留普通翻转等基础变换。
- EMA=True，AMP=True（启动命令需带 `--use-amp`），正常 COCO bbox 评估，不做坐标平移校准。

40 轮每轮约 337 次优化器更新，共约 13,480 次。相对于全量训练 120 轮，训练图像
处理量约为 1/30；这不是实际耗时保证，完整验证、I/O、增强和通信也占时间。

主指标用 AP@[.50:.95]，同时看 AP75、APs、六类 AP、训练速度和显存。
短程结果相近时暂不排位，先比较基线在种子 0/1 的波动。只提升少数类的一次偶然
结果、只有 AP50 提升或收敛更慢，都不能直接作为最终胜负。慢收敛方法可进入
80 轮确认，避免被短程筛选误杀。正式结论仍需要全量成对实验。

## Git 更新与数据生成

在服务器执行。先查看工作区；已有无关未跟踪配置保留。若发现已跟踪修改或分叉，
先检查具体内容，不使用 reset 或强制 checkout。

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
SUBSET=/root/zxy/experiments/data_subsets/sardet100k_screen_v1
if [ ! -f "$SUBSET/stats.json" ]; then
  python tools/dataset/prepare_sardet_screen.py --output-dir "$SUBSET"
fi
cat "$SUBSET/stats.json"
```

生成的数据和统计不进入 Git。脚本仅用 Python 标准库；训练使用 deim_zxy。

## 启动 40 轮基线

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
CONFIG=configs/deim_dfine/deim_hgnetv2_s_sardet10pct_screen_40e.yml
SEED=0
OUT=$(mktemp -d /root/zxy/experiments/deim_s_sardet_screen40e_seed${SEED}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_XXXXXX)
export OUT
git rev-parse HEAD > "$OUT/git_revision.txt"
cp "$CONFIG" "$OUT/config.yml"
cp /root/zxy/experiments/data_subsets/sardet100k_screen_v1/stats.json "$OUT/subset_stats.json"
printf '%s\n' "$CONFIG" > "$OUT/config_path.txt"
printf '%s\n' "$SEED" > "$OUT/seed.txt"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$CONFIG" -t /root/zxy/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --seed "$SEED" --use-amp --output-dir "$OUT" \
  > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
printf 'OUT=%s\nPID=%s\n' "$OUT" "$PID"
```

80 轮入围确认使用完全相同的启动块，将 CONFIG 改成
`configs/deim_dfine/deim_hgnetv2_s_sardet10pct_confirm_80e.yml`，OUT 的名称改成
`deim_s_sardet_confirm80e_seed...`，依次用 SEED=0 和 SEED=1。基线与候选都这样
从 COCO 初始化；方法修改另建继承配置，保留相同数据、预算和增强。

## 日志、指标与 TensorBoard

重连时先激活环境，并把 OUT 设置成启动命令输出的实际目录。

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/这里替换为实际实验目录
export OUT
tail -f "$OUT/console.log"
```

另一个终端可查看进程和 GPU：

```bash
OUT=/root/zxy/experiments/这里替换为实际实验目录
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
watch -n 2 nvidia-smi
```

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/这里替换为实际实验目录
export OUT
python - <<'PY'
import json, os, pathlib
p=pathlib.Path(os.environ['OUT'])
rows=[json.loads(s) for s in (p/'log.txt').read_text().splitlines() if s.strip()]
best=max(rows,key=lambda r:r['test_coco_eval_bbox'][0])
for label,r in [('best',best),('latest',rows[-1])]:
    print(label,'epoch',r['epoch'],dict(zip(['AP','AP50','AP75','APs','APm','APl'],
          [round(100*x,3) for x in r['test_coco_eval_bbox'][:6]])))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

## 选择 best 权重、完整验证及六类 AP

按权重中记录的 epoch 对应的 AP 选择 best_stg1/best_stg2，不能只看文件名。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/这里替换为实际实验目录
export OUT
CKPT=$(python - <<'PY'
import json, os, pathlib, sys, torch
p=pathlib.Path(os.environ['OUT'])
rows={r['epoch']:r for s in (p/'log.txt').read_text().splitlines() if s.strip() for r in [json.loads(s)]}
choices=[]
for name in ['best_stg1.pth','best_stg2.pth']:
    f=p/name
    if f.is_file():
        epoch=torch.load(f,map_location='cpu',weights_only=False)['last_epoch']
        choices.append((rows[epoch]['test_coco_eval_bbox'][0],f))
ap,f=max(choices,key=lambda x:x[0])
print('selected',f,'AP',100*ap,file=sys.stderr)
print(f)
PY
)
CONFIG=$(cat "$OUT/config_path.txt")
EVAL=$(mktemp -d "$OUT/fullval_XXXXXX")
export EVAL
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$CONFIG" -r "$CKPT" --test-only --use-amp --output-dir "$EVAL" \
  -u val_dataloader.dataset.ann_file=/root/zxy/TCRA-SAR/datasets/SARDet_100K/Annotations/val.json \
  > "$EVAL/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$EVAL/launcher.pid"
disown "$PID"
printf 'EVAL=%s\n' "$EVAL"
tail -f "$EVAL/console.log"
```

完整验证完成后，读取本次对应 best 权重的六类 AP：

```bash
export EVAL=/root/zxy/experiments/实际实验目录/fullval_实际后缀
python - <<'PY'
import os, pathlib, torch
d=torch.load(pathlib.Path(os.environ['EVAL'])/'eval.pth',map_location='cpu',weights_only=False)
for k,name in enumerate(['ship','aircraft','car','tank','bridge','harbor']):
    values=d['precision'][:,:,k,0,-1]
    valid=values[values>=0]
    print(name,'AP',round(float(valid.mean())*100,3) if valid.size else 'N/A')
PY
```

训练中的 `eval/latest.pth` 对应最近一次评估，不等同于 best 权重的评估。
epoch 指标写入 `log.txt`，TensorBoard 写入 `summary/`；COCO 指标 0、1、2 分别为
AP、AP50、AP75，原值乘 100 后才是表格中常用的百分数。

## 恢复同一训练

当前 solver 只在无强增强阶段前更新 last.pth：40 轮协议最后更新到 epoch 35，
80 轮协议更新到 epoch 71。它不是最终轮权重。恢复使用原配置、原输出目录和
`-r`，不再带 `-t`；进入最后阶段后中断可能需要从此前保存点重复部分轮次。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/这里替换为实际实验目录
CONFIG=$(cat "$OUT/config_path.txt")
SEED=$(cat "$OUT/seed.txt")
RESUME="$OUT/last.pth"
RESUME_LOG="$OUT/resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).console.log"
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$CONFIG" -r "$RESUME" --seed "$SEED" --use-amp --output-dir "$OUT" \
  > "$RESUME_LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
tail -f "$RESUME_LOG"
```

已验证真实标注的抽样配额、固定种子复现、训练子集内容保持一致、六类覆盖及两个
配置的继承/日程一致性。服务器生成后还会确认选中图像路径可读。未启动模型训练，
没有声称该短程协议已经证明能预测全量方法排名或实际墙钟耗时。
