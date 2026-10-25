# 第 0 步：FDR 分布方差是否携带可用的边定位信息（证伪实验）

不训练、不改损失、不碰调度。在现有 baseline checkpoint 上导出 FDR 的四边分布，
测量其方差与真实边误差的关系。目的是在投入模块开发之前，先判断方向是否成立。

## 为什么做这个，而不是直接写模块

仓库此前的记录显示，SCAS / SBR / DPM / BCM / NWD / HWD 都是先实现再验证，全部无收益。
其中 SBR 的失败有明确解释：`docs/localization_diagnosis_2026-10-24.md` 的受控实验测出
前后景分离的上限只有 +1.05 AP（AP 59.36 → 60.41），该方向已被测量否定。

同一份诊断给出的可用空间是：

| variant | AP | 相对 AP 的增量 |
|---|---|---|
| AP | 59.36 | — |
| AP_GTBOX | 92.37 | +33.01（定位精度上限） |
| AP_IQS | 64.25 | +4.89（按真实 IoU 排序的上限） |
| AP_TPFP | 60.41 | +1.05（前后景分离上限） |

本实验检验的假设是：D-FINE 的 FDR 已经为每条边算了 33 个 bin 的完整分布，
但推理时只取期望值，分布形状被丢弃；这个被丢弃的二阶信息，能否用来逼近 AP_IQS 那部分空间。

如果测不出关系，本方向立即终止，节省后续数周。

## 代码事实（已核对，带行号）

1. **只取均值。** `engine/deim/dfine_decoder.py:267-271` 的 `Integral.forward`：
   softmax 后直接 `F.linear(x, project)` 求期望，方差与形状全部丢弃。

2. **唯一的分布形状使用者是 LQE，且当前近似恒等。**
   `dfine_decoder.py:274-289`：只用 `prob.topk(4)` 的峰值与峰均值，不看方差；
   末尾 `init.constant_(...bias, 0)` / `init.constant_(...weight, 0)` 将输出层初始化为零，
   因此训练起点上 LQE 是恒等映射，与代码注释 "Lqe does not affect the performance here" 一致。

3. **推理不输出分布。** `dfine_decoder.py:823-825`：eval 分支只输出
   `pred_logits` 与 `pred_boxes`；`pred_corners` 与 `ref_points` 仅在 training 分支输出。

4. 参数：`reg_max=32`（33 个 bin），`up=0.5`，`reg_scale=4`，`num_layers=3`，`eval_idx=-1`。

结论：要拿到方差，必须让 eval 分支把 `pred_corners` 和 `ref_points` 一起输出。
这是本实验唯一的模型侧改动。

---

## 第 1 步：eval 分支追加两个输出

修改 `engine/deim/dfine_decoder.py` 的第 823-825 行区域：

```python
        if self.training:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'pred_corners': out_corners[-1],
                   'ref_points': out_refs[-1], 'up': self.up, 'reg_scale': self.reg_scale}
        else:
            out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1],
                   # Diagnosis only: FDR bin logits and the reference point that
                   # distance2bbox decoded them from. Consumers that used to read
                   # only pred_logits/pred_boxes are unaffected.
                   'pred_corners': out_corners[-1], 'ref_points': out_refs[-1],
                   'up': self.up, 'reg_scale': self.reg_scale}
```

安全性：`engine/deim/postprocessor.py:51` 只读 `pred_logits` 与 `pred_boxes`；
`engine/solver/det_engine.py:141` 只读 `pred_boxes`。追加键不影响正常评估。
`up` 与 `reg_scale` 是 `requires_grad=False` 的 Parameter，导出时取 `.item()`。

不要修改别的逻辑。这一改动只增加输出，不改变任何数值。

## 第 2 步：导出逐边分布统计

新建 `tools/export_corner_stats.py`。不要改 `tools/generate_predictions.py`，
基线导出链路保持原样以便复现。

```python
#!/usr/bin/env python3
"""
Export per-edge FDR distribution statistics for the localization analysis.

The decoder computes 4 x (reg_max+1) bin logits per query and Integrals them to
a mean. The spread around that mean is thrown away in every existing code path,
so it is unknown whether it carries usable information about edge error. This
tool exports, per query, the mean and standard deviation of each edge
distribution, together with the box they decode to and the reference point used.

Usage:
python tools/export_corner_stats.py -c <config> -r <checkpoint> -o corners.npz \
    --score-thr 0.05 --topk 100
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from pycocotools.coco import COCO
from tqdm import tqdm

from engine.core import YAMLConfig
from engine.deim.dfine_utils import weighting_function


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cfg.model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint['ema']['module'] if checkpoint.get('ema') is not None else checkpoint['model']
    model.load_state_dict(state)
    model.eval()

    reg_max = int(model.decoder.reg_max)
    up = model.decoder.up.detach().float().cpu()
    reg_scale = float(model.decoder.reg_scale.detach().cpu())
    project = weighting_function(reg_max, up, reg_scale).float().cpu()

    out_img, out_qidx = [], []
    out_score, out_label = [], []
    out_box, out_ref = [], []
    out_mean, out_std = [], []
    recon_err = 0.0
    n_img = 0

    loader = cfg.val_dataloader
    with torch.no_grad():
        for samples, targets in tqdm(loader, desc='export'):
            samples = samples.to(device)
            outputs = model(samples)

            corners = outputs['pred_corners'].detach().float().cpu()   # [B, L, 4*(reg_max+1)]
            refs = outputs['ref_points'].detach().float().cpu()        # [B, L, 4] cxcywh
            boxes = outputs['pred_boxes'].detach().float().cpu()       # [B, L, 4] cxcywh
            logits = outputs['pred_logits'].detach().float().cpu()

            B, L, D = corners.shape
            prob = torch.softmax(corners.reshape(B, L, 4, reg_max + 1), dim=-1)
            mean = (prob * project.view(1, 1, 1, -1)).sum(-1)
            var = (prob * (project.view(1, 1, 1, -1) - mean.unsqueeze(-1)) ** 2).sum(-1)
            std = var.clamp(min=0).sqrt()

            # Anchor check: the mean must rebuild pred_boxes through distance2bbox.
            # A large discrepancy means pred_corners/ref_points do not describe pred_boxes.
            rs = abs(reg_scale)
            x1 = refs[..., 0] - (0.5 * rs + mean[..., 0]) * (refs[..., 2] / rs)
            y1 = refs[..., 1] - (0.5 * rs + mean[..., 1]) * (refs[..., 3] / rs)
            x2 = refs[..., 0] + (0.5 * rs + mean[..., 2]) * (refs[..., 2] / rs)
            y2 = refs[..., 1] + (0.5 * rs + mean[..., 3]) * (refs[..., 3] / rs)
            rec = torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], -1)
            recon_err = max(recon_err, (rec - boxes).abs().max().item())
            n_img += B

            scores, labels = logits.sigmoid().max(dim=-1)               # [B, L]

            for i in range(B):
                image_id = int(targets[i]['image_id'].item())
                s = scores[i]
                keep = (s > args.score_thr).nonzero(as_tuple=True)[0]
                if keep.numel() == 0:
                    continue
                if keep.numel() > args.topk:
                    keep = keep[s[keep].topk(args.topk).indices]
                out_img.append(np.full(keep.numel(), image_id, dtype=np.int64))
                out_qidx.append(keep.numpy().astype(np.int32))
                out_score.append(s[keep].numpy())
                out_label.append(labels[i][keep].numpy().astype(np.int32))
                out_box.append(boxes[i][keep].numpy())
                out_ref.append(refs[i][keep].numpy())
                out_mean.append(mean[i][keep].numpy())
                out_std.append(std[i][keep].numpy())

    def cat(x, dtype=None):
        if not x:
            return np.zeros((0,), dtype=dtype or np.float32)
        out = np.concatenate(x, 0)
        return out if dtype is None else out.astype(dtype)

    np.savez_compressed(
        args.output,
        image_id=cat(out_img, np.int64),
        query_idx=cat(out_qidx, np.int32),
        score=cat(out_score),
        label=cat(out_label, np.int32),
        box=cat(out_box).reshape(-1, 4),
        ref=cat(out_ref).reshape(-1, 4),
        mean=cat(out_mean).reshape(-1, 4),
        std=cat(out_std).reshape(-1, 4),
        project=project.numpy(),
        reg_scale=np.float64(reg_scale),
        reg_max=np.int64(reg_max),
    )

    print(f"images: {n_img}")
    print(f"records: {len(cat(out_img))}")
    print(f"max |mean-decoded box - pred_boxes| = {recon_err:.6e}")
    if recon_err > 1e-4:
        print("WARNING: reconstruction mismatch; pred_corners do not describe pred_boxes.")
        print("         Do not interpret the distribution statistics until this is resolved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--resume', required=True)
    parser.add_argument('-o', '--output', default='corners.npz')
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--topk', type=int, default=100)
    args = parser.parse_args()
    main(args)
```

`recon_err` 是本工具的自检：用 `mean` 走 `distance2bbox` 的公式重建框，必须与
`pred_boxes` 一致（< 1e-4）。不一致说明 `pred_corners` 与 `pred_boxes` 不同源，
后续任何统计都无意义，必须先解决。

## 第 3 步：执行导出

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
RUN=<替换为实际基线运行目录>
DIAG="$RUN/diagnosis"
mkdir -p "$DIAG"
export CUDA_VISIBLE_DEVICES=0
python tools/export_corner_stats.py \
  -c "$RUN/resolved_config.yml" \
  -r "$RUN/best_stg1.pth" \
  -o "$DIAG/corners.npz" \
  --score-thr 0.05 --topk 100
```

若最佳检查点是 `best_stg2.pth`，替换 `-r`。4096 张验证图、单卡，预计十几分钟。

## 第 4 步：分析

新建 `tools/corner_signal_analysis.py`：

```python
#!/usr/bin/env python3
"""
Measure whether the spread of the FDR edge distributions predicts edge error.

For each matched (detection, GT) pair the four edges are compared in the same
unit the bins use, so the statistics are directly comparable to the softmax
spread:

  d = (ref_cx - x1) / (ref_w / reg_scale) - 0.5 * reg_scale     (left)

Three questions, in the order that decides whether a module is worth building:

  1. signal        corr(sigma_edge, |error_edge|) per edge, overall and by size
  2. anisotropy    are the four edges' sigma genuinely different?
  3. calibration   does mean |error| rise monotonically with sigma?

Usage:
python tools/corner_signal_analysis.py --npz corners.npz --gt val.json \
    --iou-thr 0.5 --output analysis.txt
"""

import argparse
import json
from collections import defaultdict

import numpy as np

EDGE = ['left', 'top', 'right', 'bottom']


def iou_matrix(a, b):
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    ua = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    out = np.zeros_like(inter)
    ok = ua > 0
    out[ok] = inter[ok] / ua[ok]
    return out


def edge_dists(ref, boxes, reg_scale):
    """Bin-space edge distances, matching dfine_utils.bbox2distance."""
    rs = abs(reg_scale)
    cx, cy, w, h = ref[:, 0], ref[:, 1], ref[:, 2], ref[:, 3]
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    left = (cx - x1) / (w / rs + 1e-16) - 0.5 * rs
    top = (cy - y1) / (h / rs + 1e-16) - 0.5 * rs
    right = (x2 - cx) / (w / rs + 1e-16) - 0.5 * rs
    bottom = (y2 - cy) / (h / rs + 1e-16) - 0.5 * rs
    return np.stack([left, top, right, bottom], -1)


def corr(a, b):
    if len(a) < 30 or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--border', type=float, default=0.02,
                    help='Exclude GT boxes this close to an image edge (normalised); '
                         'clipping there fakes an edge asymmetry.')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    z = np.load(args.npz)
    reg_scale = float(z['reg_scale'])
    reg_max = int(z['reg_max'])
    gt = json.load(open(args.gt))

    img = {im['id']: im for im in gt['images']}
    gts = defaultdict(list)
    for a in gt['annotations']:
        if a.get('iscrowd', 0):
            continue
        gts[a['image_id']].append(a)

    dets = defaultdict(list)
    for i in range(len(z['image_id'])):
        dets[int(z['image_id'][i])].append(i)

    W_gt, SIG, ERR = [], [], []
    for image_id, idxs in dets.items():
        cand = gts.get(image_id, [])
        if not cand:
            continue
        gboxes = np.array([g['bbox'] for g in cand], np.float64)
        glabels = np.array([g['category_id'] for g in cand])
        im = img.get(image_id, {})
        W_img, H_img = im.get('width', 0), im.get('height', 0)

        idxs = sorted(idxs, key=lambda k: -z['score'][k])
        used = np.zeros(len(cand), dtype=bool)

        for k in idxs:
            box = z['box'][k]                      # normalised cxcywh
            ref = z['ref'][k]
            lab = z['label'][k]
            # back to pixel xywh for IoU, same convention as generate_predictions
            px = np.array([(box[0] - box[2] / 2) * W_img, (box[1] - box[3] / 2) * H_img,
                           box[2] * W_img, box[3] * H_img])
            m = glabels == lab
            if not m.any():
                continue
            cand_g = np.where(m)[0]
            ious = iou_matrix(px, gboxes[cand_g])[0]
            j = int(np.argmax(ious))
            if ious[j] < args.iou_thr or used[cand_g[j]]:
                continue
            used[cand_g[j]] = True
            g = gboxes[cand_g[j]]
            if W_img and H_img and args.border > 0:
                if (g[0] / W_img < args.border or g[1] / H_img < args.border or
                        (g[0] + g[2]) / W_img > 1 - args.border or
                        (g[1] + g[3]) / H_img > 1 - args.border):
                    continue
            gbox = np.array([(g[0] + g[2] / 2) / W_img, (g[1] + g[3] / 2) / H_img,
                             g[2] / W_img, g[3] / H_img])
            d_pred = edge_dists(ref[None], box[None], reg_scale)[0]
            d_gt = edge_dists(ref[None], gbox[None], reg_scale)[0]
            if (d_gt < 0).any() or (d_gt > reg_max - 0.1).any():
                continue                            # target outside bin range
            W_gt.append(g[2])
            SIG.append(z['std'][k])
            ERR.append(d_pred - d_gt)

    if not SIG:
        print('no matched detections')
        return
    SIG = np.array(SIG)
    ERR = np.array(ERR)
    W_gt = np.array(W_gt)
    lines = []

    def p(s):
        print(s)
        lines.append(s)

    p(f"matched pairs: {len(SIG)}")
    p("\n== 1. signal: corr(sigma, |error|) per edge ==")
    p(f"{'edge':8s} {'n':>7s} {'corr':>8s} {'mean_sigma':>11s} {'mean|err|':>10s}")
    for e, name in enumerate(EDGE):
        s, a = SIG[:, e], np.abs(ERR[:, e])
        p(f"{name:8s} {len(s):7d} {corr(s, a):8.3f} {s.mean():11.4f} {a.mean():10.4f}")

    p("\n== 1b. same, by GT size ==")
    bins = [('tiny<32', 0, 32), ('small32-96', 32, 96), ('med96-256', 96, 256), ('large>256', 256, 1e9)]
    for bn, lo, hi in bins:
        sel = (W_gt >= lo) & (W_gt < hi)
        if sel.sum() < 30:
            p(f"{bn:12s} n={sel.sum():6d}  (too few)")
            continue
        cs = [corr(SIG[sel, e], np.abs(ERR[sel, e])) for e in range(4)]
        p(f"{bn:12s} n={sel.sum():6d}  corr L/T/R/B = " +
          " ".join(f"{c:6.3f}" if c == c else "   nan" for c in cs))

    p("\n== 2. anisotropy: is any edge systematically less certain? ==")
    p("   (excludes boxes near the image border, where clipping fakes asymmetry)")
    mu = SIG.mean(0)
    p(f"mean sigma  L={mu[0]:.4f} T={mu[1]:.4f} R={mu[2]:.4f} B={mu[3]:.4f}")
    p(f"ratio       L/R={mu[0]/max(mu[2],1e-9):.3f}  T/B={mu[1]/max(mu[3],1e-9):.3f}")
    p(f"spread      max/min={mu.max()/max(mu.min(),1e-9):.3f}")
    p("   RandomHorizontalFlip p=0.5 during training averages away any true left/right")
    p("   asymmetry, so L/R similarity is expected. A top/bottom difference is the one")
    p("   that can survive augmentation and is the one with physical meaning in SAR.")

    p("\n== 3. calibration: mean |error| by sigma quantile ==")
    for e, name in enumerate(EDGE):
        s, a = SIG[:, e], np.abs(ERR[:, e])
        q = np.quantile(s, [0, .2, .4, .6, .8, 1.0])
        p(f"  {name}:")
        for i in range(5):
            sel = (s >= q[i]) & (s <= q[i + 1])
            if sel.sum() < 20:
                continue
            p(f"    q{i+1} sigma[{q[i]:.4f},{q[i+1]:.4f}] n={sel.sum():6d} "
              f"mean|err|={a[sel].mean():.4f} corr={corr(s[sel], a[sel]):6.3f}")

    p("\n== 4. combined signal vs IoU ==")
    # A single scalar per detection: mean sigma across edges, compared with the
    # achieved IoU. If this is flat, the spread carries no ranking information.
    p("see step 1 for the per-edge version; a scalar summary is only meaningful")
    p("if at least one edge shows corr >= 0.3 above.")

    if args.output:
        open(args.output, 'w').write('\n'.join(lines) + '\n')
        print(f"\nwritten {args.output}")


if __name__ == '__main__':
    main()
```

执行：

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
RUN=<替换为实际基线运行目录>
DIAG="$RUN/diagnosis"
python tools/corner_signal_analysis.py \
  --npz "$DIAG/corners.npz" \
  --gt /root/zxy/experiments/data_subsets/sardet100k_screen_v1/val.json \
  --iou-thr 0.5 \
  --output "$DIAG/corner_signal.txt"
cat "$DIAG/corner_signal.txt"
```

纯 CPU，几秒到一分钟。

前置检查：本脚本用 `label`（模型输出的 0..N−1）与 GT 的 `category_id` 直接比对。
基线配置为 `remap_mscoco_category: False`，因此 `val.json` 的 category_id 必须是
从 0 开始的训练标签。若两者不一致，所有匹配会静默失败、输出 `no matched detections`。
先确认：

```bash
python - /root/zxy/experiments/data_subsets/sardet100k_screen_v1/val.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(sorted({c['id'] for c in d['categories']}))
print(sorted({a['category_id'] for a in d['annotations']}))
PY
```

两组都应是 `[0, 1, 2, 3, 4, 5]`。

## 第 5 步：同时跑定位上限诊断

这套方案的天花板取决于 SARDet 上 `AP_IQS` 相对 AP 的增量。OGSOD 上是 +4.89，
SARDet 上可能不同。**这个数字必须先拿到**，否则模块做出来收益上限未知。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
RUN=<替换为实际基线运行目录>
DIAG="$RUN/diagnosis"
python tools/generate_predictions.py \
  -c "$RUN/resolved_config.yml" -r "$RUN/best_stg1.pth" \
  -o "$DIAG/pred.json" --score-thr 0.001 --no-nms
python tools/localization_ceiling.py \
  --gt "$DIAG/pred_gt.json" --dt "$DIAG/pred.json" \
  --iou-thr 0.5 --output "$DIAG/ceiling.txt"
cat "$DIAG/ceiling.txt"
```

注意 `tools/generate_predictions.py` 的坐标轴问题：若验证集图像不是全部正方形，
先按 `docs/sardet_offset_check.md` 第 1 步修正 `orig_size` 索引，否则该诊断的
AP 数值不可信。`localization_ceiling` 只比较同一批检测在不同条件下的相对变化，
即便有缩放误差，`AP_GTBOX` 与 `AP_IQS` 的**相对大小**仍可参考，但 AP 绝对值不可引用。

---

## 判读标准

按顺序看，任一条不通过则终止，不进入模块开发。

| # | 指标 | 通过 | 存疑 | 终止 |
|---|---|---|---|---|
| 0 | `recon_err` | < 1e-4 | — | > 1e-4：分布与框不同源，先解决 |
| 1 | 四条边 `corr(sigma, |err|)` 的中位数 | ≥ 0.30 | 0.15–0.30 | < 0.15 |
| 2 | 小目标档（tiny<32）的相关性 | 高于整体 | 持平 | 更低 |
| 3 | 边间 `max/min` sigma 比 | ≥ 1.15 | 1.05–1.15 | < 1.05 |
| 4 | 校准单调性 | mean|err| 随 sigma 分位单调上升 | 有抖动 | 无趋势或反向 |
| 5 | `AP_IQS − AP` | ≥ 3.0 | 1.5–3.0 | < 1.5 |

补充说明：

- 第 1 项是关键。方差与误差不相关，说明分布的宽度不反映定位难度，方向即终止。
- 第 3 项决定论文能否立住。若四边 sigma 基本相等（各向同性），方案退化为
  "再预测一次定位质量"，与 GFL 的 LQE 无本质区别，新颖性不足。
  但 L/R 相似是**预期内**的（水平翻转增强会抹平真实左右差异），
  所以真正要看的是 **T/B 的差异**。
- 第 4 项单调性比相关系数更能说明信号可用：模块需要一个可标定的量，不能只是弱相关。
- 第 5 项是上限。若 `AP_IQS − AP` 在 SARDet 上只有 1 左右，本方向的天花板就低，
  应该换方向而不是做当前这个。

## 交付要求

回填到本文件：

- `recon_err` 数值。
- 第 1 项四条边的相关系数与样本数。
- 第 2 项分尺寸的相关系数表。
- 第 3 项四边 sigma 均值、L/R 与 T/B 比值、max/min。
- 第 4 项校准表。
- 第 5 项 `AP`、`AP_GTBOX`、`AP_IQS`、`AP_T PFP` 与各自增量。
- 按判读表给出结论：进入模块设计 / 换方向 / 终止，并说明依据是第几项。

未执行的步骤写「未执行」及原因。

## 若通过，模块设计草案（暂不实现）

三部分，全部在当前张量上操作，不新增骨干、不改特征金字塔：

1. **边级不确定性头**：从 `pred_corners` 的 softmax 直接算 4 维 std（无参数），
   或经一个轻量 MLP 映射。无参版本更干净，先试它。
2. **逐边标定损失**：回归 `|err_edge|`，四条边分别监督，**不要塌缩成标量 IoU**——
   一旦塌缩就退化成 LQE，丢掉方向性。建议用 bin 空间单位，与分布同一尺度。
3. **分数修正**：`logits + g(std)`，g 初始化为零（与现有 LQE 同策略，保证起点不退化）。

消融三条：只加 1（看分布本身有多少信息）、加 1+2（看标定是否有效）、加 1+2+3（看 AP/AP75）。

风险留档：若第 3 项显示各向同性，第 1、2 部分的动机消失，此时应改为直接做第 3 部分，
并把论文定位从"SAR 边缘物理"改为"标定过的定位质量排序"，新颖性会显著下降，
需要重新评估是否值得投入。
