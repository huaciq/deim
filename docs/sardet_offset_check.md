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

---

## 执行记录：2026-10-25，用户指定的已完成旧基线

### 运行、检查点和诊断口径

用户在执行过程中指定先诊断以下已完成的运行：

```text
/root/zxy/experiments/deim_s_sardet10pct_baseline120e_b32_fp16retry_seed0_20261024_224350_WjEvpc
```

该运行有完整 120 条 epoch 指标，最后日志 epoch=119。它是此前的旧协议：
global batch=32、GPU 0/1、seed=0、640 基础/验证输入、COCO 初始化、EMA/FP16，
FP32 loss 与异常批次 FP32 重算开启；AdamW 主 LR=4e-4、骨干 LR=2e-4、WD=1e-4，
该运行沿用旧调度，实际 **warmup=1000、flat=60、no_aug=8**，
增强边界 **[4,60,112]**、MixUp=[4,60]，Mosaic/MixUp 概率均为 0.5。
输入训练集 9,449 张、验证集 4,096 张；RandomZoomOut/RandomIoUCrop 关闭。
因此本记录不能当作后来 warmup=2000、flat=64、stop=118 新协议的最终结果。
本次没有启动、恢复或停止任何训练，也未修改模型、训练配置或原始标注。

训练代码 revision=`70113b4`；本次导出代码 revision=`9aa5dd2`。
后者只修正 `tools/generate_predictions.py` 四处宽高索引及对应注释。

| 阶段 | 按日志 AP 选择的 epoch | AP | AP50 | AP75 | APs |
| --- | ---: | ---: | ---: | ---: | ---: |
| stage1，epoch<112 | 111 | 42.6836 | 72.5333 | 44.3151 | 35.5818 |
| stage2，epoch>=112 | **118** | **42.7546** | **72.6263** | **44.5886** | **35.7815** |

选用 **best_stg2.pth**，不是按阶段文件名盲选。该日志记录的 APm=50.3777、APl=43.1035。
检查点已复制为诊断目录中的 `checkpoint_best_stg2.pth`，固定本次输入。

```text
DIAG=/root/zxy/experiments/deim_s_sardet10pct_baseline120e_b32_fp16retry_seed0_20261024_224350_WjEvpc/diagnosis_offset_20261025_200358_OZrXwK
```

该目录保存冻结配置、尺寸分布、元数据、预测、GT、所有分析输出及执行脚本。
导出使用物理 GPU 1、单图 batch、FP32，以控制与在跑训练共享 GPU 时的显存占用；
只在诊断副本 `export_config.yml` 中将验证 batch 改为 1，原运行配置保持不变。
score_thr=0.001、无 NMS、seed=0，共导出 **1,196,316** 条预测。

服务器没有独立安装 pycocotools；首次直接启动因此未完成导出，记录保留为
`export_attempt1.console.log`。后续使用训练环境已有的 **faster-coco-eval 1.8.0**，
先调用 `init_as_pycocotools()` 再运行原分析工具，不安装依赖、不修改其它导出逻辑。

### 第 0、1 步：尺寸分布及坐标修正

- 验证图 4,096 张；width/height 缺失 **0**；正方形 **4,047**，非正方形 **49**。
- 不同宽高对 **51**；宽度 min/median/max=224/256/1000，高度也为 224/256/1000。
- 常见尺寸：256×256：2,815；512×512：827；800×800：382；224×224：18；
  1000×1000：5。其余 49 张为非正方形。
- 因存在非正方形图，已执行第 1 步：按 orig_size=[W,H]，x 乘 W、y 乘 H，
  与训练评估 PostProcessor 对齐。500×300、300×500、256×256 的实际导出表达式均
  与 torchvision box_convert 加 [W,H,W,H] 缩放结果一致。
- 另只读确认 OGSOD 验证标注的 3,667 张图全部为 256×256。本次索引修正不会改变
  那批正方形图的历史导出数值，不能解释其已记录的 +0.79px。

### 第 3 步：AP 锚点及诊断上限

日志最佳 AP=**42.754596**，导出重算 AP≈**42.88**，差约 **+0.1254 AP**，
通过本方案规定的 |差值|≤0.2 门槛。文本 AP 输出保留两位小数。

这个锚点是容差核对，不是逐检测完全一致：原导出器对每个 query 取类别最大分，
训练后处理对 query×class 展平后取 top-300，且导出应用 score 阈值。
此次按要求保留这些逻辑。下表及平移扫描统一以 **导出 AP≈42.88** 为参考，
不能将其替代真实 baseline 的 42.7546，也不能将 oracle 增量直接加到日志 AP 上。

| 变换 | AP | AP50 | AP75 | APs | 相对导出 AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| AP | 42.88 | 72.73 | 44.81 | 35.88 | 0 |
| AP_GTBOX | 72.72 | 72.73 | 72.72 | 67.48 | +29.84 |
| AP_TPFP | 47.78 | 88.45 | 47.15 | 39.71 | +4.90 |
| AP_IQS | 50.94 | 81.88 | 53.84 | 42.78 | +8.06 |
| AP_BOTH | 88.45 | 88.45 | 88.45 | 81.35 | +45.57 |

这些是依赖 GT 的受控变换，不是可实现方法的真实收益，几个增量不能相加。
定位精度存在较大改善空间，排序与误检也有改善空间，不能沿用 OGSOD 上的具体比例。

### 第 4 步：四个分组的偏移测量

同类别、IoU≥0.5 的贪心匹配共 **8,324** 对。全局
**dx=+0.194px（sd=3.475）**、**dy=−0.185px（sd=3.123）**，均为源图像像素。

| 类别 | n | dx 均值 | dx sd | dy 均值 | dy sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| ship | 3985 | +0.059 | 2.649 | −0.008 | 2.530 |
| aircraft | 2016 | +0.102 | 4.807 | −0.572 | 4.413 |
| car | 485 | +0.648 | 1.886 | −0.284 | 2.349 |
| tank | 677 | +0.287 | 1.303 | −0.251 | 0.990 |
| bridge | 1035 | +0.521 | 3.789 | −0.041 | 2.704 |
| harbor | 126 | +1.053 | 7.873 | +0.003 | 6.326 |

| 图像宽度 | n | dx 均值 | dy 均值 |
| --- | ---: | ---: | ---: |
| 224 | 182 | +0.312 | −0.196 |
| 256 | 4472 | +0.254 | −0.027 |
| 500 | 26 | +0.408 | −1.242 |
| 512 | 2457 | +0.221 | −0.549 |
| 800 | 1096 | −0.160 | −0.073 |
| 1000 | 31 | +0.829 | +2.680 |

按原工具规则省略 n<20 的图宽组，共涉及 60 对匹配；500、1000 组样本也较少。

下表同时回填 per GT width 与 offset scaling；最后一列是逐匹配对 dx/W_gt 的均值，
不是组均值 dx 除以某个代表宽度。

| GT 宽度（源像素） | n | dx 均值 | dy 均值 | mean(dx/W_gt) |
| --- | ---: | ---: | ---: | ---: |
| <12 | 1026 | +0.238 | −0.060 | +0.0294 |
| 12–24 | 1742 | +0.193 | −0.029 | +0.0126 |
| 24–48 | 2656 | +0.291 | −0.136 | +0.0089 |
| 48–96 | 2354 | +0.136 | −0.375 | +0.0026 |
| ≥96 | 546 | −0.102 | −0.329 | +0.0004 |

补充位置分组：最低 x 分位组 dx=+0.915，最高 x 分位组 dx=−0.647；
混合图宽下拟合为 dx=−0.00192*x+0.586，残差 sd=3.461。
这只是相关性，混合了图像尺寸、类别和目标大小，不能仅凭该回归判定归一化代码错误。

**判定：没有复现 OGSOD 式统一 +0.79px 的 x 平移、y≈0 的模式。**
同时也不满足“所有组都小于 0.2px”，不能宣称完全没有偏移。
本检查点存在较小且分组相关的双轴残差，不支持一个全局共享的常量平移解释。
类别间不同也不足以单独证明标注错误；本次不对数据集、标注或模型内部作排他性归因。

### 第 5 步：平移的 AP 代价

因部分分组存在明显残差，执行了原脚本的两段扫描，未修改搜索逻辑。

| 搜索 | 最佳 dx | 最佳 dy | AP | AP75 | APs | oracle ΔAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 粗扫 range=2、step=0.5 | −0.50 | 0 | 43.03 | 44.89 | 35.82 | +0.15 |
| 细扫 range=1、step=0.25 | **−0.25** | **0** | **43.13** | **45.00** | **36.21** | **约 +0.26** |

AP 和 ΔAP 分别独立四舍五入，因此末位可能有 0.01 的差别。
粗扫最佳点在内部，细扫范围覆盖该点；**扩大到 range=4、step=1 未执行**，
因为没有边界最优，按方案无需扩大。每次均为 9 个 dx、9 个 dy，另评估一次原始 AP。

这约 0.26 AP 是验证集上已知 GT 的 oracle 诊断值，远小于 OGSOD 的 5.12 AP。
未将平移应用到模型、训练或正式评估，也不将其作为论文方法收益。

### 第 6 步：逐段定位及中心/尺寸误差

按要求原样执行 pipeline_roundtrip_test.py，日志保存在 pipeline_roundtrip.console.log。
DEIM resize、两次/单次 hflip、归一化及 255→不同尺寸往返均 OK。
但其 torchvision Resize 一项报 dx=−21.2、dy=−19 的 OFFSET。

该项测试使用的是**普通 Tensor 框**；torchvision v2 不把它当作带图像尺寸的框处理，
所以图像缩放而框未缩放。实际 CocoDetection 使用 convert_to_tv_tensor 创建
BoundingBoxes，这个测试输入与生产管线不一致。故不能把该 OFFSET 当作训练 Resize 的缺陷。
原脚本也没有实际执行其说明中承诺的 Mosaic 放置测试。

因此额外保存并执行 pipeline_production_probe.py，未修改原脚本或训练代码：

- 使用生产 BoundingBoxes 类型，对 256×256、500×300、300×500 的 torchvision Resize
  和两次水平翻转验证，坐标误差均为 0。
- 调用实际 Mosaic 的样本缩放、四象限放置、恒等仿射和最终 Resize，坐标误差均为 0。
- 实际 ConvertBoxes 归一化后逆变换的最大绝对坐标误差为 0.0000611px，
  均值 dx≈0.0000010px、dy≈0.0000114px。

已测确定性坐标操作没有解释观测残差。此检查未穷尽随机仿射、训练动态和标注差异，
因此不能沿用原脚本末尾“全部 OK 就必然来自模型内部”的因果结论。
明确定位并修复的是**导出工具 W/H 用反**；没有定位到生产训练管线中的固定偏移缺陷。

| 源图 GT 面积档 | n | 原 IoU | 修正尺寸后的 IoU_c | 修正中心后的 IoU_s | 尺寸修正增益 | 中心修正增益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 8324 | 0.784 | 0.838 | 0.835 | +0.053 | +0.051 |
| <32² | 4134 | 0.750 | 0.804 | 0.818 | +0.053 | +0.068 |
| 32²–96² | 3791 | 0.821 | 0.873 | 0.854 | +0.052 | +0.033 |
| 96²–256² | 388 | 0.795 | 0.854 | 0.831 | +0.059 | +0.036 |
| ≥256² | 11 | 0.715 | 0.775 | 0.738 | +0.060 | +0.023 |

按两个单因素 IoU 增益归一化，总体约**尺寸 51%、中心 49%**；<32² 档约
**尺寸 44%、中心 56%**。这只是增益占比，不是严格可加的误差百分比。
≥256² 仅 11 对，不据此推断普遍结论。总体中心和尺寸都需要改进，未发现单一因素
独占定位误差；<32² 档中心误差相对更值得观察。

### 复现、产物与 Git 交接

原始大体积预测、GT、权重快照及诊断日志只保存在上述 DIAG，未提交到 Git。
本文件记录必要汇总；本地完整交接和小型原始输出副本位于
`C:/Users/DELL/.codex/artifacts/sardet_offset_check_20261025/`。
`dimensions.json`、`diagnosis_metadata.json`、`anchor.json` 记录口径，
`ceiling.txt`、`bias_locator.txt`、`position_dependence.txt`、`shift_coarse.txt`、
`shift_fine.txt`、`box_decomp.txt` 与对应 console.log 保存实际测量。
初次缺失依赖和初次未设置 PYTHONPATH 的补充检查日志也保留，没有覆盖成功结果。

服务器源代码已通过 Git bundle 的 fetch、--ff-only merge 同步。正常联网时更新命令为：

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
ulimit -Sn 65535
RUN=/root/zxy/experiments/deim_s_sardet10pct_baseline120e_b32_fp16retry_seed0_20261024_224350_WjEvpc
DIAG="$RUN/diagnosis_offset_20261025_200358_OZrXwK"
```

以下只复现诊断，建立新目录，避免覆盖本次产物：

```bash
REPRO=$(mktemp -d "$DIAG/reproduce_XXXXXX")
coco_run() {
  local SCRIPT="$1"
  shift
  python -c "import faster_coco_eval; faster_coco_eval.init_as_pycocotools(); import runpy,sys; script=sys.argv.pop(1); runpy.run_path(script,run_name='__main__')" "$SCRIPT" "$@"
}
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2
coco_run tools/generate_predictions.py \
  -c "$DIAG/export_config.yml" -r "$DIAG/checkpoint_best_stg2.pth" \
  -o "$REPRO/pred_sardet.json" --score-thr 0.001 --no-nms --seed 0
coco_run tools/localization_ceiling.py \
  --gt "$REPRO/pred_sardet_gt.json" --dt "$REPRO/pred_sardet.json" \
  --iou-thr 0.5 --output "$REPRO/ceiling.txt"
```

先核对 AP 与 42.7546 的差值不超过 0.2，超出则停止排查，不继续后面的分析。
锚点通过后：

```bash
coco_run tools/bias_locator.py \
  --gt "$REPRO/pred_sardet_gt.json" --dt "$REPRO/pred_sardet.json" \
  --iou-thr 0.5 --output "$REPRO/bias_locator.txt"
coco_run tools/bias_position_dependence.py \
  --gt "$REPRO/pred_sardet_gt.json" --dt "$REPRO/pred_sardet.json" \
  --iou-thr 0.5 --output "$REPRO/position_dependence.txt"
coco_run tools/bias_shift_search.py \
  --gt "$REPRO/pred_sardet_gt.json" --dt "$REPRO/pred_sardet.json" \
  --range 2.0 --step 0.5 --output "$REPRO/shift_coarse.txt"
coco_run tools/bias_shift_search.py \
  --gt "$REPRO/pred_sardet_gt.json" --dt "$REPRO/pred_sardet.json" \
  --range 1.0 --step 0.25 --output "$REPRO/shift_fine.txt"
coco_run tools/pipeline_roundtrip_test.py | tee "$REPRO/pipeline_roundtrip.txt"
PYTHONPATH=/root/zxy/deim python "$DIAG/pipeline_production_probe.py" \
  | tee "$REPRO/pipeline_production_probe.txt"
coco_run tools/box_error_decomposition.py \
  --gt "$REPRO/pred_sardet_gt.json" --dt "$REPRO/pred_sardet.json" \
  --iou-thr 0.5 --output "$REPRO/box_decomp.txt"
```

监控本次产物：

```bash
cat "$DIAG/analysis_status.json"
cat "$DIAG/anchor.json"
tail -n 8 "$DIAG/fine_shift.console.log"
cat "$DIAG/bias_locator.txt"
cat "$DIAG/box_decomp.txt"
watch -n 2 nvidia-smi
```

本次不训练。旧协议从 COCO 重新训练、恢复、检查点评估及 TensorBoard 的完整命令见
[旧协议交接](sardet_fp16_retry_restart.md)，重连后先激活 deim_zxy、设置 OUT，
并在启动命令前执行 `ulimit -Sn 65535`；使用 -t 初始化、-r 恢复，二者不同时传入。
查看该已完成运行的原始日志/指标：

```bash
OUT="$RUN"
tail -n 8 "$OUT/console.log"
tail -n 3 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```

第 0–6 步均已执行；扩大搜索范围按内部最优条件跳过，其原因已在上文记录。
