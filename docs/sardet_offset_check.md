# SARDet-100K 子集：系统框偏移验证

目的：在 SARDet-100K 六类筛选子集的 120e/b32 基线上，验证 OGSOD 阶段观测到的
「预测框相对 GT 存在 x 向系统偏移」是否存在。本方案只做诊断，不启动训练、不改模型。

OGSOD 侧已记录的事实（`tools/bias_locator.py`、`tools/bias_shift_search.py` 文档字符串）：

- 已匹配 (pred, GT) 对的 `dx = pred_cx - gt_cx` 均值约 +0.79 源像素，`dy` 约 -0.001；
  每个尺寸档、每个类别方向一致。
- 全局平移 dx=-0.75、dy=0 可让 AP 提升 5.12。
- `sd(w_pred/w_gt) = 0.104`（小目标），即随机散布，与上述系统偏差是两个不同量。
- 从未记录 `tools/pipeline_roundtrip_test.py` 的运行结果，因果关系未定位。

OGSOD 结论不自动迁移到 SARDet。本方案给出判定方法。

---

## 阻塞问题：预测导出工具的坐标轴反了

`tools/generate_predictions.py:91-96`：

```python
cx, cy, w, h = boxes_keep.unbind(-1)
boxes_xyxy = torch.stack([
    (cx - 0.5 * w) * orig_size[1],
    (cy - 0.5 * h) * orig_size[0],
    (cx + 0.5 * w) * orig_size[1],
    (cy + 0.5 * h) * orig_size[0]
], dim=-1)
```

`orig_size` 来自 `engine/data/dataset/coco_dataset.py:174` 的
`torch.as_tensor([int(w), int(h)])`，即 `[W, H]`。因此上面 x 乘以 H、y 乘以 W。

评估用的 `engine/deim/postprocessor.py:55` 是
`bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)`，即 `[W, H, W, H]`，
x 乘 W、y 乘 H。两者不一致。

影响：W == H 的图上无差异；W != H 时导出坐标错误，误差为 `cx * (H - W)`，
正比于 x 位置，表现为缩放误差而非常量平移。

OGSOD 的图像若为正方形，该缺陷是惰性的，不解释其 +0.79px 常量偏移。
SARDet-100K 图像尺寸多样，该缺陷必然生效，不修会污染本次全部测量。

**先做第 0 步：若验证集全部为正方形，跳过第 1 步。**

---

## 第 0 步：图像尺寸分布

只读 JSON，不需 GPU。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
VAL_ANN=/root/zxy/experiments/data_subsets/sardet100k_screen_v1/val.json
python - "$VAL_ANN" <<'PY'
import json, sys
from collections import Counter
d = json.load(open(sys.argv[1]))
ims = d['images']
print(f"images: {len(ims)}")
missing = [i['id'] for i in ims if 'width' not in i or 'height' not in i]
print(f"missing width/height: {len(missing)}")
if missing:
    print("  WARNING: per-image-width analysis in bias_locator will be meaningless")
sq = sum(1 for i in ims if i.get('width') == i.get('height'))
print(f"square images: {sq} / {len(ims)}")
wh = Counter((i.get('width'), i.get('height')) for i in ims)
print(f"distinct (w,h) pairs: {len(wh)}")
for (w, h), n in wh.most_common(10):
    print(f"  {w}x{h}: {n}")
ws = [i['width'] for i in ims if 'width' in i]
hs = [i['height'] for i in ims if 'height' in i]
if ws:
    print(f"width  min/median/max: {min(ws)}/{sorted(ws)[len(ws)//2]}/{max(ws)}")
    print(f"height min/median/max: {min(hs)}/{sorted(hs)[len(hs)//2]}/{max(hs)}")
PY
```

判读：

- `square images` 等于总数 → 导出缺陷惰性，跳过第 1 步。
- 否则 → 必须执行第 1 步。
- `missing width/height > 0` → `bias_locator` 的 per-image-width 分组失效，
  「像素常量 vs 相对量」无法判定，需先补全标注文件的 width/height。

---

## 第 1 步：修正导出坐标（非正方形时必做）

`tools/generate_predictions.py` 中，把四处 `orig_size` 索引改为：

```python
                boxes_xyxy = torch.stack([
                    (cx - 0.5 * w) * orig_size[0],
                    (cy - 0.5 * h) * orig_size[1],
                    (cx + 0.5 * w) * orig_size[0],
                    (cy + 0.5 * h) * orig_size[1]
                ], dim=-1)
```

不要改动其他逻辑。修正后建议补一行注释说明 `orig_size` 为 `[W, H]`，与
postprocessor 的 `[W, H, W, H]` 对齐。

历史上在 OGSOD 上导出的 predictions 与本次修正后不可直接比较。若 OGSOD 图像为正方形，
两者数值一致；若不是，旧的分析需要重跑。这一点在论文中不体现，但要记录在案。

---

## 第 2 步：导出预测

先确定基线运行目录与最佳检查点。不要假定路径，从运行目录已保存的配置读取。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
RUN=/root/zxy/experiments/deim_s_sardet10pct_paper120_b32_nozoom_nocrop/seed0   # 改成实际目录
ls "$RUN"
cat "$RUN/resolved_config.yml" 2>/dev/null | head -20 || cat "$RUN"/*config*.yml | head -20
grep -c . "$RUN/log.txt"
```

按评估 AP 选检查点，不要只看文件名。`best_stg1.pth` 只覆盖 `epoch < 118`，
`best_stg2.pth` 覆盖 `epoch >= 118`（`engine/solver/det_solver.py:139-152`），
真实最佳可能在后一个里：

```bash
python - "$RUN/log.txt" <<'PY'
import json, sys
best = None
for i, line in enumerate(open(sys.argv[1])):
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    m = d.get('test_coco_eval_bbox')
    if not m:
        continue
    if best is None or m[0] > best[1]:
        best = (d.get('epoch', i), m[0], m)
if best is None:
    print("no test_coco_eval_bbox found")
else:
    e, ap, m = best
    f = 'best_stg2.pth' if e >= 118 else 'best_stg1.pth'
    print(f"best epoch={e} AP={m[0]*100:.2f} AP50={m[1]*100:.2f} "
          f"AP75={m[2]*100:.2f} APs={m[3]*100:.2f} APm={m[4]*100:.2f} APl={m[5]*100:.2f}")
    print(f"checkpoint file: {f}")
PY
```

若 `best_stg2.pth` 不存在（最后两轮未刷新记录），用 `best_stg1.pth`。

导出（口径与 OGSOD 诊断一致：低阈值、不 NMS）：

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
RUN=/root/zxy/experiments/deim_s_sardet10pct_paper120_b32_nozoom_nocrop/seed0
CKPT="$RUN/best_stg1.pth"
CFG="$RUN/resolved_config.yml"
DIAG="$RUN/diagnosis"
mkdir -p "$DIAG"
export CUDA_VISIBLE_DEVICES=0
python tools/generate_predictions.py \
  -c "$CFG" \
  -r "$CKPT" \
  -o "$DIAG/pred_sardet.json" \
  --score-thr 0.001 --no-nms
```

产出 `$DIAG/pred_sardet.json` 与 `$DIAG/pred_sardet_gt.json`。

若 `resolved_config.yml` 不存在，改用运行目录中的 `source_config.yml`，
或直接从仓库配置 `configs/deim_dfine/deim_hgnetv2_s_sardet10pct_baseline_120e_b32_paper_nozoom_nocrop.yml`
启动，但必须先核对其 `img_folder` / `ann_file` 与实际数据位置一致。

---

## 第 3 步：锚点核对（关键自检）

导出的预测重算 AP，必须与 `log.txt` 中同一检查点的 AP 一致。不一致则后续分析全部无效，
停止并排查导出链路。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
RUN=/root/zxy/experiments/deim_s_sardet10pct_paper120_b32_nozoom_nocrop/seed0
DIAG="$RUN/diagnosis"
python tools/localization_ceiling.py \
  --gt "$DIAG/pred_sardet_gt.json" \
  --dt "$DIAG/pred_sardet.json" \
  --iou-thr 0.5 \
  --output "$DIAG/ceiling.txt"
head -20 "$DIAG/ceiling.txt"
```

`ceiling.txt` 中的 `AP` 一项应与第 2 步从 `log.txt` 读出的 AP 相差在 0.2 以内
（差异来自 score 阈值与 NMS 设置，属预期）。差距更大说明导出有问题。

同时该文件给出 SARDet 上的四类上限：`AP_GTBOX`、`AP_TPFP`、`AP_IQS`、`AP_BOTH`。
这是判断「定位是否仍是主瓶颈」的直接依据，比沿用 OGSOD 的 88.4% 更可靠。

---

## 第 4 步：偏移量测量

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
RUN=/root/zxy/experiments/deim_s_sardet10pct_paper120_b32_nozoom_nocrop/seed0
DIAG="$RUN/diagnosis"
python tools/bias_locator.py \
  --gt "$DIAG/pred_sardet_gt.json" \
  --dt "$DIAG/pred_sardet.json" \
  --iou-thr 0.5 \
  --output "$DIAG/bias_locator.txt"
cat "$DIAG/bias_locator.txt"
```

输出四个分组：per category、per image width、per GT width、offset scaling。

判读表：

| 观测 | 结论 | 下一步 |
| --- | --- | --- |
| dx ≈ 0 且 dy ≈ 0（各档均 < 0.2px） | SARDet 上无此问题；OGSOD 的偏移是其自身数据/管线问题 | 跳过第 5、6 步；定位工作按 SARDet 重新做 |
| dx 为固定像素常量，不随 GT 宽度和图像宽度变化 | 坐标约定缺陷，全局共享 | 第 5 步计量，第 6 步定位 |
| dx 正比于 x 位置或图像宽度 | 导出或归一化环节缺陷 | 检查第 1 步是否已修；再跑第 6 步 |
| dx 正比于 GT 宽度 | 相对偏移 | 属模型或标注，第 6 步仅作参考 |
| 只有部分类别有偏移 | 标注问题，指向具体类别 | 不要当作管线缺陷修 |

注意区分轴向：OGSOD 侧是 x 有偏、y 精确为零。若 SARDet 是两轴同时有偏，
性质不同，不能套用同一解释。

---

## 第 5 步：偏移的 AP 代价（仅在第 4 步测出明显偏移时执行）

默认 `--range 2.0 --step 0.25` 是 17×17 = 289 次 COCOeval，在 4096 张图上耗时不可接受。

该工具已经内置两段扫描：先扫 dx（dy=0），再在最优 dx 固定下扫 dy，
因此**只跑一次即可，不要改脚本**。粗扫一次：

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
RUN=/root/zxy/experiments/deim_s_sardet10pct_paper120_b32_nozoom_nocrop/seed0
DIAG="$RUN/diagnosis"
python tools/bias_shift_search.py \
  --gt "$DIAG/pred_sardet_gt.json" \
  --dt "$DIAG/pred_sardet.json" \
  --range 2.0 --step 0.5 \
  --output "$DIAG/shift_coarse.txt"
cat "$DIAG/shift_coarse.txt"
```

`--range 2.0 --step 0.5` 是 9 个 dx 加 9 个 dy 共 18 次评估，可接受。
若最优格点落在 ±2.0 边界上，说明真实偏移超出搜索范围，改用 `--range 4.0 --step 1.0` 重来；
若最优格点落在内部，再用 `--range 1.0 --step 0.25` 在其附近细化一次。

注意：得到的 AP 增量是在验证集上、已知 GT 时的 oracle 数值，**只用于诊断，
不可写入论文作为结果**。它估计的是「修好这个偏差能拿回多少」，不等于修复后的真实收益，
因为源头修复会同时改变训练信号。

---

## 第 6 步：定位到具体环节

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
python tools/pipeline_roundtrip_test.py | tee /tmp/roundtrip.txt
```

该脚本逐段隔离测试 resize 往返、torchvision Resize、hflip 两次与单次镜像约定、
normalize cxcywh 往返、Mosaic 摆放。输出中任何 `**OFFSET**` 直接定位到该段。

若全部 OK，脚本末尾的提示是：偏移来自模型内部（参考点或网格构造），而非数据管线。
此时再执行：

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
RUN=/root/zxy/experiments/deim_s_sardet10pct_paper120_b32_nozoom_nocrop/seed0
DIAG="$RUN/diagnosis"
python tools/box_error_decomposition.py \
  --gt "$DIAG/pred_sardet_gt.json" \
  --dt "$DIAG/pred_sardet.json" \
  --iou-thr 0.5 \
  --output "$DIAG/box_decomp.txt"
cat "$DIAG/box_decomp.txt"
```

该输出区分中心误差与尺寸误差（`IoU_c` / `IoU_s`），是决定后续做「位置证据」还是
「尺度估计」的依据。

---

## 交付要求

执行完回填以下内容到本文件末尾，并提交：

- 第 0 步的尺寸分布结论（是否为正方形、distinct 宽高对数）。
- 基线检查点对应 epoch 与 AP/AP50/AP75/APs（此前仓库中无记录）。
- 第 3 步锚点核对的 AP 偏差。
- 第 4 步四个分组的 dx/dy 均值，以及按上表得到的判定。
- 若执行了第 5、6 步，给出 oracle AP 增量、定位到的环节、中心/尺寸误差占比。

未执行的步骤明确写「未执行」及原因，不要留空。
