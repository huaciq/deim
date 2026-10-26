# SAR-AIRcraft baseline 分析复现

分析工具不修改模型、训练配置、原标注或检查点。训练曲线使用完整 `log.txt`，
最佳模型由 val AP 选择。test 只统计标注尺寸，不进行模型推理或调参。

## 本次运行与分析口径

- 训练目录：`/root/zxy/experiments/deim_m_sar_aircraft_102e_b28_nozoom_nocrop_seed0_20261026_142643_LMw1Yy`。
- 训练 revision：`a448005`；102 轮，GPU 0/1，全局 batch 28，seed 0。
- 配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_102e_b28.yml`；其余协议见
  [M 固定协议](sar_aircraft_deim_m_protocol.md)。
- 日志逻辑 epoch 90 最佳，选 `best_stg2.pth` 的 EMA；内部 `last_epoch=87` 来自
  官方收尾阶段回载 best_stg1，不等于逻辑轮次。`last.pth` 仍是 epoch 89 的恢复点。
- val 固定 640、FP32、原后处理 top-300、不加 NMS、不预先过滤置信度。
- 完整预测导出用每卡 batch 8；日志中的原验证用每卡 batch 32。独立复算 AP
  相比原日志低 0.002726 个百分点，两者分开记录；同一导出预测用原 faster-coco-eval
  与 pycocotools 得到相同指标。
- 主错误阈值固定 score≥0.3、IoU≥0.5；补充 score=0.1/0.5，未用阈值搜索改 AP。
- 先按分数匹配尚未配对的同类 GT，再把 FP 互斥分为重复、分类、定位、分类与定位同时错、背景。
  “背景”是相对现有 GT 的 IoU<0.1 几何定义，不是所有区域都经人工确认没有真实目标。
- 漏检按 GT 计数，FP 按预测计数，分母不同；计数不等于 TIDE dAP。
- 网络尺度按原始 bbox 直接缩放至 480/640/800 计算，未重建 Mosaic 后的训练框分布。
  640 尺度 AP 用同时缩放 GT/预测后的 COCO area-range 评估，保留框外忽略规则。

## 服务器通过 Git 更新

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
OUT=/root/zxy/experiments/deim_m_sar_aircraft_102e_b28_nozoom_nocrop_seed0_20261026_142643_LMw1Yy
DATA=/root/zxy/datasets/SAR-AIRcraft-1.0
```

有未提交或分叉时先查看，不 reset、不强制 checkout。GitHub HTTPS 不通时，本次实际
通过 Git bundle 上传、`git fetch /tmp/...bundle HEAD`、`git merge --ff-only FETCH_HEAD`
更新工具。数据已经准备好，不重新生成标注。

## 双卡导出与案例库

重跑分析创建新目录，避免覆盖已交付的分析结果：

```bash
ANA=$(mktemp -d "$OUT/analysis_val_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_XXXXXX")
export OUT DATA ANA
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 \
  tools/export_analysis_predictions.py -c "$OUT/resolved_config.yml" -r "$OUT/best_stg2.pth" \
  --ann-file "$DATA/COCO/val.json" --batch-per-gpu 8 --output "$ANA/predictions.json.gz" \
  > "$ANA/export.console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$ANA/export.pid"
disown "$PID"
tail -f "$ANA/export.console.log"
```

导出完成后运行 CPU 图像特征与案例生成，只需已有 numpy/Pillow：

```bash
nohup python -u tools/build_sar_case_library.py --annotations "$DATA/COCO/val.json" \
  --predictions "$ANA/predictions.json.gz" --image-root "$DATA" --output "$ANA/cases" \
  > "$ANA/cases.console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$ANA/cases.pid"
disown "$PID"
tail -f "$ANA/cases.console.log"
```

`tools/analyze_sar_baseline.py` 需要 numpy、matplotlib、Pillow、pycocotools。
本次在 Windows Conda `pytorch` 生成统计与曲线（numpy 1.26.4、matplotlib 3.9.2、
Pillow 11.0.0），服务器环境缺少绘图包；结果归档回服务器，不为此改训练环境。
在已具备这些依赖的 Linux 分析环境可直接执行：

```bash
python tools/analyze_sar_baseline.py --source "$OUT" --annotations "$DATA/COCO" \
  --predictions "$ANA/predictions.json.gz" --output "$ANA/report" \
  --run "$OUT" --revision a448005
```

将 `cases/` 放到 `report/cases/` 后打开 `report/report.md` 和 `report/cases/index.html`。
每个案例保留 image_id、gt_id、预测类别/分数和图像代理数值；现象标签需看图解释，
不能把 CNR 代理等同于物理可见度。人工视觉检查记录与自动候选清单分别保存。

## 训练复现与恢复命令

本任务只分析已完成训练；以下用于将来复现，不自动启动第二次训练：

```bash
CONFIG=configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_102e_b28.yml
REPRO=$(mktemp -d /root/zxy/experiments/deim_m_sar_aircraft_repro_seed0_XXXXXX)
cp "$OUT/resolved_config.yml" "$REPRO/resolved_config.yml"
git rev-parse HEAD > "$REPRO/git_revision.txt"
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$REPRO/resolved_config.yml" -t /root/zxy/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth \
  --seed 0 --use-amp --output-dir "$REPRO" --summary-dir "$REPRO/summary" \
  > "$REPRO/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$REPRO/launcher.pid"
disown "$PID"
```

只针对已中断且停止的训练恢复，`last.pth` 是强增强停止前的恢复点，
不是本次最终或最佳模型；`-r` 与 COCO 初始化 `-t` 不同时提供：

```bash
RLOG="$OUT/resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).console.log"
nohup env CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/last.pth" --seed 0 --use-amp \
  --output-dir "$OUT" --summary-dir "$OUT/summary" > "$RLOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
```

## 日志、指标与进程

重新连接时重新激活环境、设置 `OUT`/`ANA`，本次实际 `ANA="$OUT/analysis_val"`。

```bash
tail -f "$OUT/console.log"
tail -n 3 "$OUT/log.txt"
tail -f "$ANA/export.console.log"
tail -f "$ANA/cases.console.log"
ps -p "$(cat "$ANA/export.pid")" -o pid,etime,args
watch -n 2 nvidia-smi
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

训练的 JSONL bbox 数组 0/1/2 分别是 AP/AP50/AP75，8 是 AR100，原始值需乘 100。
分析 `report/summary.json`、CSV 与曲线均留存，不提交标注、预测、权重或图片到 Git。
