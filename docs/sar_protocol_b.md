# SAR-AIRcraft 主协议 B-v1：trainval 训练、统一尺度 1024、固定日程

日期：2026-11-03。依据：[协议 v2 与 A/B 方案](sar_protocol_v2.md)、[文献协议调研](sar_aircraft_protocol_survey.md)、
[型号混淆线索诊断](sar_confusion_cues_diagnosis.md)。旧协议（640 冻结、方案 A）结果全部保留，作为诊断与对照。

## 1. 协议

| 项目 | 设定 | 依据 |
|---|---|---|
| 训练集合 | 官方 trainval（`COCO/trainval.json`，3489 张，13513 个目标） | 官方数据集论文"训练集 13513 / 测试集 2950"、SAR-Net"默认划分" |
| 报告 | 官方 test（879 张）AP50 为主，另报 AP、AP75、各类 AP50；val 已在训练集内，不作为指标 | 与官方、SAR-Net 同口径；test 有 77% 目标在 trainval 中出现，属数据集本身性质，论文中说明 |
| 选权重 | 不选：`fixed_schedule: True`，取最后一轮 `final.pth` | val 在训练集内；640 运行中 final 与 val 选权重在 test 上一致 |
| 输入 | 每张图右/下补边到 1500×1500 后缩放到 1024：所有图同一缩放比 0.683；测试相同 | 数据集 1 m 分辨率，同类飞机在各尺寸源图中像素尺寸一致；直接缩放会使缩放比相差最多 1.9 倍 |
| 多尺度 | 保留 DEIM 批级多尺度 768–1280（同一 batch 内所有图同一尺度），第 120 轮停止 | 既往单种子对比 64.5 对 58.7（有/无多尺度）；批级缩放不破坏图间尺度一致 |
| 增强 | 只保留水平翻转 + DEIM mixup（第 4–70 轮）；关闭 Mosaic、PhotometricDistort、ZoomOut、IoUCrop | Mosaic 对每张图单独缩放到固定尺寸并随机 0.5–1.5 缩放，破坏统一尺度；SAR 单通道，色调/饱和度无效，对比度抖动改变斑点统计；ZoomOut/IoUCrop 改尺度且黑边不像杂波 |
| 初始化 | COCO M 权重（`-t`），7 个类别头全部由 COCO airplane 一行初始化 | 原 80→7 映射失败，类别头随机；单卡前 3 轮 AP 0.1/2.3/3.4 → 4.8/7.7/10.6 |
| 日程 | 132 轮：warmup 600 iter（约 2 轮，二次曲线），flat 到第 70 轮，cosine 到第 120 轮，最后 12 轮无增强 | 原 2000 iter 二次 warmup 约 18 轮、半程时 LR 仅峰值 26%，前 15 轮几乎不学；1024 下 102 轮未收敛 |
| 优化 | AdamW，主 LR 2.3e-4、骨干 2.3e-5，WD 1e-4，全局 batch 12（每卡 6），EMA 0.9999 | 1024 实测每卡 6 张峰值 13.4 GB；EMA 后期稳定高 1.3–1.5 AP（640 checkpoint 实测） |
| 精度 | AMP FP16 + FP32 loss/retry | 沿用 |
| 模型 | DEIM-D-FINE-M，seed 0 | 沿用 |

配置：`configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_b_canvas1024_132e_b12_old_server.yml`。
实现：`PadToCanvas`（`engine/data/transforms/_transforms.py`，补边后把 `orig_size` 设为画布尺寸，预测框按画布映射回原图像素；
左上对齐，原图坐标不变）；`tools/init_head_from_coco_class.py`（生成类别头初始化权重）；solver 的 `fixed_schedule` 与 `final.pth`。

**已知局限与预期**

- 统一尺度让 A330/A320 等尺寸差异大的类别更容易区分；但诊断显示主要混淆对（737/787、A220/ARJ21）的源图像素尺寸本身重叠
  （可分性 0.51–0.64），这一项对它们的帮助预计有限。
- 800 图补边后有效区域只占画布 28%，计算有浪费；这是统一尺度的代价。
- 一次同时改了多项设定，作为"协议"整体与旧配方比较；单项贡献如需写入论文，再在此协议上做去掉单项的消融。
- 预计每轮约 2.6 分钟（291 iter），132 轮约 5.5–6 小时；显存与已测 1024 运行相同（张量尺寸一致）。

## 1.1 B-v1 baseline 结果（seed 0）

运行：`/home/zxy/sar/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_b_canvas1024_132e_b12_old_server_seed0_20261103_193451_xwONF9`，
`final.pth`（epoch 131）在官方 test 上评估，输出在该目录的 `test_eval_final/`。

| AP | AP50 | AP75 | APs | APm | APl | AR100 |
|---:|---:|---:|---:|---:|---:|---:|
| 68.09 | 87.91 | 74.64 | 61.4 | 67.6 | 67.3 | 81.4 |

| 类别 | AP | AP50 | AP75 |
|---|---:|---:|---:|
| A220 | 65.79 | 87.45 | 66.71 |
| A320/321 | 81.28 | 100.00 | 95.38 |
| A330 | 84.50 | 95.95 | 95.95 |
| ARJ21 | 60.22 | 81.08 | 63.89 |
| Boeing737 | 61.39 | 87.62 | 61.98 |
| Boeing787 | 62.22 | 80.01 | 73.22 |
| other | 61.23 | 83.27 | 65.32 |

按预定判定（test AP50 ≥ 约 83）冻结 B-v1：之后的方法和消融都在这组配置上比较，不再按 test 调配方。

判读：

- 与只用 train 的旧配方（640，AP 45.74 / AP50 65.37）相比 AP +22.35、AP50 +22.54，但同时改了训练集合和配方，
  两者各占多少未单独测量（"旧配方 + trainval"对照决定不跑）。预计训练集合是主要来源：test 中 train+val 见过的目标占 77%（只用 train 时 50%），
  737 从 23% 升到 72%，其 AP 从约 23 升到 61.39。
- 第 131 轮 train_loss 与 val AP 仍缓慢变化（val 在训练集内，只反映拟合），不能据此判断 test 是否收敛；不用 test 选轮数，日程固定为 132 轮。
- 单种子。test 去重后约 381 架飞机，且 77% 目标在训练集中有近似副本，模块比较需要多种子，并按见过/未见过拆开看。

## 2. 运行顺序

基线命名为 DEIM-D-FINE-M（D-FINE 架构 + DEIM 的 MAL、MixUp 与 COCO 权重），不因后续结果更改。

1. B-v1 baseline（本配置，已完成）。
2. B-v1 + REM。
3. 按错误类型拆解 B-v1（混淆、漏检、定位，见过/未见过），据此设计方法（斑点重采样、细粒度分类等）。
4. B-v1 补 seed 1、2，得到种子波动。
5. 最优组合跑 3 个种子；补 YOLOv8、plain D-FINE-M 对照（只作对照，不用于更换基线）。

"旧配方 + trainval"对照不跑：它同时差多项设定，不能归因，也不改变任何决定。单项贡献如需写入论文，在 B-v1 上做去掉单项的消融。

目标：多种子均值的 test AP50 超过 SAR-Net 的 90.2。val 在训练集内，模块只能在 test 上比较，因此候选提前列定，失败的也如实报告。

## 3. 服务器命令（旧服务器）

更新代码与环境：

```bash
cd /home/zxy/sar/repos/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deim_zxy
export DATA=/home/zxy/sar/datasets/SAR-AIRcraft-1.0
```

一次性准备：确认 trainval 标注（应为 3489 张、13513 个目标），生成类别头初始化权重（新文件，不覆盖原权重）：

```bash
python -c "import json; d=json.load(open('$DATA/COCO/trainval.json')); print(len(d['images']), len(d['annotations']))"
python tools/init_head_from_coco_class.py /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e.pth \
  /home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e_airplane7.pth
```

启动（两卡空闲时）：

```bash
nvidia-smi -i 0,1
export EXPERIMENT_ROOT=/home/zxy/sar/experiments
export INITIAL_WEIGHTS=/home/zxy/sar/pretrain/deim/deim_dfine_hgnetv2_m_coco_90e_airplane7.pth
bash tools/run_sar_module_experiment.sh configs/deim_dfine/deim_hgnetv2_m_sar_aircraft_b_canvas1024_132e_b12_old_server.yml
```

启动后检查（重连后同样先设置 `OUT`、激活环境）：

```bash
export OUT=$(ls -dt /home/zxy/sar/experiments/deim_m_sar_aircraft_deim_hgnetv2_m_sar_aircraft_b_canvas1024_132e_b12_old_server_* | head -n 1); echo "$OUT"
grep -m1 "fixed_schedule" "$OUT/console.log"
grep -m1 "Load model.state_dict" "$OUT/console.log" | cut -c1-400
grep -E "Epoch: \[[0-9]+\]" "$OUT/console.log" | tail -n 1 | grep -oE "Epoch: \[[0-9]+\] +\[[0-9]+/[0-9]+\]|eta: [0-9:]+|time: [0-9.]+|data: [0-9.]+|max mem: [0-9]+"
grep -iE "out of memory|Traceback" "$OUT/console.log" | tail -n 3
```

`Load model.state_dict` 一行里 `unmatched` 应为空（类别头尺寸与 7 类一致，已加载）。每轮应为 291 iter。

监控：

```bash
tail -f "$OUT/console.log"
ps -p "$(cat "$OUT/launcher.pid")" -o pid,etime,args
nvidia-smi -i 0,1
python - "$OUT/log.txt" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
for r in rows[-3:]:
    v = r['test_coco_eval_bbox']
    print(r['epoch'], 'train_loss', round(r['train_loss'], 3), '(val in trainval) AP', round(v[0] * 100, 2), 'AP50', round(v[1] * 100, 2))
PY
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

只在进程已停止时恢复：

```bash
bash tools/run_sar_module_experiment.sh --resume "$OUT"
```

完成（日志最后为 epoch 131，出现 `final.pth`）后在 test 上评估：

```bash
ls -lh "$OUT"/final.pth
export FINALEVAL="$OUT/test_eval_final"; mkdir "$FINALEVAL"
CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 nohup torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/final.pth" --test-only --seed 0 \
  --output-dir "$FINALEVAL" --summary-dir "$FINALEVAL/summary" \
  -u val_dataloader.dataset.img_folder="$DATA" val_dataloader.dataset.ann_file="$DATA/COCO/test.json" \
  > "$FINALEVAL/console.log" 2>&1 < /dev/null &
PID=$!; printf '%s\n' "$PID" > "$FINALEVAL/launcher.pid"; disown "$PID"
tail -f "$FINALEVAL/console.log"
grep -A12 "IoU=0.50:0.95 | area=   all | maxDets=100 ]" "$FINALEVAL/console.log" | head -n 13
```

各类 AP50：

```bash
python - "$FINALEVAL" <<'PY'
import sys, torch
from pathlib import Path
path = next(Path(sys.argv[1]).rglob('*.pth'))
ev = torch.load(path, map_location='cpu', weights_only=False)
p = ev['precision'][0, :, :, 0, -1]  # IoU 0.50, all areas, maxDets 100
names = ['A220', 'A320/321', 'A330', 'ARJ21', 'Boeing737', 'Boeing787', 'other']
for k, n in enumerate(names):
    v = p[:, k]; v = v[v > -1]
    print(n, round(float(v.mean()) * 100, 2))
PY
```

类别顺序按 `test.json` 的 categories；若数量不是 7 或顺序不同，以 `$DATA/COCO/test.json` 中的顺序为准。

## 3.1 新服务器运行 B-v1 + REM

旧服务器 GPU 被占时的单次运行，不改变默认服务器。配置 `deim_hgnetv2_m_sar_aircraft_b_canvas1024_rem_132e_b12_new_server.yml`
与旧服务器版只差路径；新服务器路径见 `AGENTS.md`，初始化权重同样由 `tools/init_head_from_coco_class.py` 生成。
B-v1 基线在旧服务器上运行，比较时注意两台机器的软件版本可能不同。

## 4. 本地验证

Windows Conda `pytorch`（CPU）：`tests/test_sar_protocol_b.py` 检查配置各项；`PadToCanvas` 对 800 与 1500 图补边后框坐标不变、
`orig_size` 为画布、缩放比一致，按画布还原到原图像素误差 <0.01 px，超出画布报错；类别头初始化复制 airplane 行并保留去噪嵌入的填充行。
未在服务器运行；torchvision 版本差异下 `F.pad` 对 PIL 与框的行为只在本地 0.29 上验证。
