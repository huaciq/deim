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

2. **LQE 使用分布峰值；零初始化只保证训练起点恒等。**
   `dfine_decoder.py:274-289`：只用 `prob.topk(4)` 的峰值与峰均值，不看方差；
   末尾 `init.constant_(...bias, 0)` / `init.constant_(...weight, 0)` 将输出层初始化为零，
   因此训练起点上 LQE 是恒等映射。训练后是否近似恒等需要实测，不能从初始化推断；
   本次检查点的 LQE 末层已非零，详见执行记录。

3. **本诊断修改前，推理不输出分布。** `dfine_decoder.py:823-825`：eval 分支只输出
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
    # weighting_function concatenates 1-D tensors; preserve the parameter shape.
    project = weighting_function(reg_max, up, model.decoder.reg_scale.detach().float().cpu()).float().cpu()

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
        recon_err=np.float64(recon_err),
        n_images=np.int64(n_img),
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

    # NpzFile does not cache arrays; read each compressed array only once.
    with np.load(args.npz) as archive:
        z = {key: archive[key] for key in archive.files}
    reg_scale = float(z['reg_scale'])
    reg_max = int(z['reg_max'])
    project_min, project_max = float(z['project'].min()), float(z['project'].max())
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
            if (d_gt < project_min).any() or (d_gt > project_max).any():
                continue                            # outside signed projection support, not index range
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

---

## 执行记录：2026-10-25，指定的新协议 baseline

### 基线、口径和改动

本次严格使用用户指定运行，不复用上一份旧基线的方框/上限诊断：

```text
RUN=/root/zxy/experiments/deim_s_sardet10pct_paper120_b32_nozoom_nocrop_fd65535_seed0_20261025_153055_ufPH9L
DIAG=$RUN/diagnosis_fdr_variance_20261025_214532_oDPpm4
```

- 已完成 120 个日志 epoch（0..119）。按 AP 选择 **best_stg2.pth、日志 epoch 119**，
  AP=**43.061709**、AP50=72.827406、AP75=45.104258、APs=35.741615，
  APm=51.015266、APl=44.951497。
- stage1 最佳为 epoch 117，AP=43.001050，低于上述 stage2，因此未选 stage1。
- 原训练：seed=0、GPU 0/1、global batch=32、基础/验证输入 640、120 轮、
  AdamW 主 LR=4e-4/骨干 LR=2e-4、WD=1e-4、warmup=2000、flat=64、
  no_aug=2，增强边界 [4,64,118]，Mosaic/MixUp 概率 0.5，关闭 ZoomOut/IoUCrop，
  COCO 检测器初始化、EMA/AMP、原 FP32 loss/异常重算策略。
- 原训练 revision=`3ef34ba`。本次唯一的模型侧改动是 eval 输出追加
  pred_corners/ref_points/up/reg_scale；没有新增可训练参数，没有修改 logits/boxes
  的计算、损失、调度或训练数据。本次未启动或恢复训练，未实现模块草案。
- 新增 tools/export_corner_stats.py、tools/corner_signal_analysis.py，保留
  generate_predictions.py 原链路不再修改。模型输出及矩计算代码 revision=`fdf07d2`，
  导出投影形状修正=`c6cc9ac`，NPZ 一次加载优化=`2ee018f`。
- 类别和标注使用的 ID 均为 [0,1,2,3,4,5]，remap_mscoco_category=False。
  验证集 4,096 张；四边导出 score>0.05、每图最多 100 queries、FP32、GPU 0。
  定位上限另用 GPU 1 导出 score>0.001、无 NMS 的预测。二者使用同一固定权重快照。

本地 4 项定向回归通过：追加输出前后 logits/boxes 逐元素一致；非均匀投影矩正确；
保留 1-D reg_scale 后生成 33 项投影；负投影目标没有被误剔除。
服务器先完成输出/矩/负目标三项回归，随后完成真实数据的全部导出与分析。
这些检查不代表新方法有效；本实验没有训练新方法。

### 执行前校正的脚本问题

上方初稿示例的以下问题已更正，实际执行以 tools/ 中的实现为准：

1. **投影值不是 bin 索引。** bin 下标为 0..32，但本检查点的投影支撑是 **[-4,4]**。
   edge_dists 返回的是连续、有符号的投影坐标，范围判断必须用 project.min/max。
   原示例的 [0,reg_max−0.1] 会错误剔除本次 6,031 个有效匹配中的 **5,546 个（91.96%）**，
   只剩 485 个，产生严重选择偏差。本报告全部使用修正后的样本。
2. weighting_function 需要保留形状为 [1] 的 reg_scale 张量来拼接投影；提前转成
   Python float 会产生零维张量拼接错误。初次导出因此退出，日志保留为
   corner_export_attempt1.console.log；修正后完整重导出，没有使用失败结果。
3. NPZ 压缩数组在分析前一次性载入，避免逐 query 反复解压。一次缓慢分析被中止后，
   用同一份 NPZ 完整重跑；没有重新采样或改变统计公式。
4. sigma 五分位区间使用互不重叠的区间，最后一区间含上界，避免分界值重复计数。

另外，**训练后的 LQE 不是零初始化状态**：最终 decoder 层的 LQE 输出层权重
L2=0.734930，bias 绝对值=0.216610。这里只确认参数已非零，未做关闭 LQE 的性能消融，
不能声称其作用近似恒等，也不能声称方差携带的信号独立于已有 LQE。

### 第 0 项：同源重建与样本数

- reg_max=32（33 bins）、up=0.5、reg_scale=4，投影支撑 [-4,4]。
- 导出 4,096 张图、**27,042** 条 query 记录。
- **recon_err=2.384185791×10⁻⁷ < 1×10⁻⁴：通过。**
  检查覆盖全部输出 queries，不只覆盖筛选后记录；均值经实际 distance2bbox 重建框。
- 同类别、IoU≥0.5 匹配 7,820 对；排除距任一图像边界不足 0.02 的 1,788 对，
  排除投影目标超出实际支撑的 1 对，最终保留 **6,031 对**。
- sigma、误差单位均为连续投影坐标，不能解释为整数 bin 个数或源图像像素。

### 第 1 项：逐边信号

| 边 | n | corr(sigma, abs(error)) | mean sigma | mean abs(error) |
| --- | ---: | ---: | ---: | ---: |
| left | 6031 | 0.45056 | 0.20462 | 0.24386 |
| top | 6031 | 0.41281 | 0.19691 | 0.23577 |
| right | 6031 | 0.43678 | 0.20671 | 0.25770 |
| bottom | 6031 | 0.42409 | 0.19887 | 0.24026 |

四边相关性的中位数为 **0.430437 ≥ 0.30：通过**。
方差确实与已匹配框的边误差相关，不能给出“方差信号不存在”的结论。

### 第 2 项：分 GT 宽度的相关系数

这里沿用方案的**源图 GT 宽度**分档，不是 COCO 按面积定义的 small/medium/large。

| GT 宽度（源像素） | n | left | top | right | bottom | 四边中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| <32 | 2879 | 0.40332 | 0.39025 | 0.38355 | 0.39943 | **0.394837** |
| 32–96 | 2836 | 0.43346 | 0.36422 | 0.47644 | 0.43692 | 0.43519 |
| 96–256 | 313 | 0.47044 | 0.52776 | 0.32813 | 0.52545 | 0.49795 |
| ≥256 | 3 | 样本不足 | 样本不足 | 样本不足 | 样本不足 | 不作判断 |

<32 档的四边相关系数均低于各自整体值，中位数 **0.394837 < 0.430437**，
按预设第 2 项落入**终止**。这不否定小目标上仍有约 0.39 的信号，
而是不支持“小目标信号强于整体”的预设动机。

### 第 3 项：四边均值与各向异性

| 指标 | 数值 |
| --- | ---: |
| mean sigma left | 0.204618 |
| mean sigma top | 0.196911 |
| mean sigma right | 0.206714 |
| mean sigma bottom | 0.198874 |
| L/R | **0.989864** |
| T/B | **0.990127** |
| max/min | **1.049783** |

四边的整体均值接近，T/B 仅相差约 1%；max/min 未达到通过阈值 1.15，
数值上略低于终止边界 1.05。**按预设第 3 项为终止**，不能把四舍五入后的 1.050
当成越过阈值。该边界不是统计显著性检验；更明确的事实是缺乏方案要求的较强
整体逐边均值差异，不能据此建立“SAR 上下边物理不确定性不同”的论断。
此结果也不等于证明每个单独目标的四边 sigma 完全相同。

### 第 4 项：五分位校准表

每条边按自身 sigma 分位，Q1..Q4 各 1,206 对，Q5 为 1,207 对。
下表为该分位内 mean abs(error)，单位为投影坐标。

| 边 | Q1（最低 sigma） | Q2 | Q3 | Q4 | Q5（最高 sigma） | 单调上升 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| left | 0.104466 | 0.154908 | 0.216130 | 0.294197 | 0.449435 | 是 |
| top | 0.100534 | 0.165008 | 0.221491 | 0.277861 | 0.413804 | 是 |
| right | 0.112446 | 0.163181 | 0.229114 | 0.324024 | 0.459579 | 是 |
| bottom | 0.093285 | 0.157174 | 0.219708 | 0.293812 | 0.437145 | 是 |

**第 4 项通过**。全部分位的 sigma 边界、计数另保存在 corner_signal.txt/json。
计算后续项目是为完成本次诊断交付，没有因第 2 项失败而进入模块开发。

补充标量相关性仅限当前匹配样本：mean(sigma) 与 IoU 的相关系数为 **−0.58313**，
与分类置信度的相关系数为 **−0.81130**。后者说明它与已有分数强相关，
尚未证明具有超出现有分类分数/LQE 的独立排序信息；未将这些相关性当作 AP 收益。

### 第 5 项：同一新基线的定位上限

使用同一 best_stg2 权重、同一 4,096 张验证图、score>0.001、无 NMS。
导出坐标已使用修正后的 W/H 约定。

日志 AP=43.061709，导出重算 AP≈43.18，差约 **+0.11829 AP**，通过 0.2 的锚点容差。
差异口径与之前一致：导出器按每 query 类别最大分取框，训练后处理对 query×class
展平后取 top-300；保留原导出逻辑。下表只以同一导出检测集合的 AP=43.18 为参照，
不替换原 baseline 的 43.061709。

| variant | AP | AP50 | AP75 | APs | 相对导出 AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| AP | 43.18 | 72.93 | 45.26 | 35.78 | 0 |
| AP_GTBOX | 72.93 | 72.93 | 72.93 | 67.25 | +29.75 |
| AP_TPFP | 47.84 | 87.95 | 47.39 | 39.86 | +4.66 |
| AP_IQS | 51.20 | 81.98 | 54.20 | 43.09 | **+8.02** |
| AP_BOTH | 87.95 | 87.95 | 87.95 | 80.86 | +44.77 |

**AP_IQS−AP=8.02≥3.0，第 5 项通过。** 这是已知 GT 的 oracle 诊断空间，
不等于 FDR 方差模块能获得 8.02 AP，几项增量也不能相加。

### 按预设标准的结论

| 项目 | 判定 |
| --- | --- |
| 0：分布/框同源重建 | 通过 |
| 1：四边相关性中位数≥0.30 | 通过，0.430437 |
| 2：小目标信号强于整体 | **终止，0.394837 更低；最先失败项** |
| 3：边间整体均值各向异性 | **终止，max/min=1.049783，T/B≈0.9901** |
| 4：分位校准单调 | 通过，四边均单调 |
| 5：排序 oracle 空间≥3 AP | 通过，+8.02 AP |

**结论：按本方案规则终止当前逐边各向异性/小目标强化模块路线，不进入模块设计或训练。**
终止依据首先是第 2 项，并有第 3 项不通过。方差与定位误差之间的关系本身成立，
所以“全部方差信息无用”不是本次结论。若以后改做一般定位质量排序，需要另行验证
相对现有 score/LQE 的独立价值；本任务未启动这个新方向，也未实现末尾模块草案。
所有要求的诊断均已执行；仅 ≥256px 档相关系数因 n=3 未作推断。

### 产物、复现与服务器交接

大体积预测、权重快照和日志仅存放在上述 DIAG，没有提交 Git。
corners.npz、corners.metadata.json、corner_signal.txt/json、ceiling.txt、anchor.json、
diagnosis_metadata.json 和本次执行脚本保留完整证据；本地小型副本在：
`C:/Users/DELL/.codex/artifacts/fdr_variance_step0_20261025/`。
首次投影类型错误及 NPZ 性能修正前的中间记录也保留，最终完成状态单独汇总。

正常服务器 Git 更新及环境激活（本次实际使用 Git bundle fetch + --ff-only merge 同步）：

```bash
cd /root/zxy/deim
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
ulimit -Sn 65535
RUN=/root/zxy/experiments/deim_s_sardet10pct_paper120_b32_nozoom_nocrop_fd65535_seed0_20261025_153055_ufPH9L
DIAG="$RUN/diagnosis_fdr_variance_20261025_214532_oDPpm4"
```

复现时创建新目录，保留本次产物；只运行推理与 CPU 分析，不训练：

```bash
REPRO=$(mktemp -d "$RUN/fdr_step0_reproduce_XXXXXX")
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python tools/export_corner_stats.py \
  -c "$DIAG/source_resolved_config.yml" -r "$DIAG/checkpoint_best_stg2.pth" \
  -o "$REPRO/corners.npz" --score-thr 0.05 --topk 100 --seed 0
python tools/corner_signal_analysis.py \
  --npz "$REPRO/corners.npz" \
  --gt /root/zxy/experiments/data_subsets/sardet100k_screen_v1/val.json \
  --iou-thr 0.5 --border 0.02 --output "$REPRO/corner_signal.txt"
coco_run() {
  local SCRIPT="$1"
  shift
  python -c "import faster_coco_eval; faster_coco_eval.init_as_pycocotools(); import runpy,sys; script=sys.argv.pop(1); runpy.run_path(script,run_name='__main__')" "$SCRIPT" "$@"
}
export CUDA_VISIBLE_DEVICES=1
coco_run tools/generate_predictions.py \
  -c "$DIAG/source_resolved_config.yml" -r "$DIAG/checkpoint_best_stg2.pth" \
  -o "$REPRO/pred.json" --score-thr 0.001 --no-nms --seed 0
coco_run tools/localization_ceiling.py \
  --gt "$REPRO/pred_gt.json" --dt "$REPRO/pred.json" \
  --iou-thr 0.5 --output "$REPRO/ceiling.txt"
```

监控与读取本次诊断：

```bash
tail -n 8 "$DIAG/corner_export.console.log"
cat "$DIAG/corner_signal.txt"
cat "$DIAG/ceiling.txt"
cat "$DIAG/anchor.json"
cat "$DIAG/final_status.json"
watch -n 2 nvidia-smi
```

以下仅交付原 baseline 的训练/恢复命令，本任务未执行。完整冻结配置、指标和
TensorBoard 交接另见 [固定协议](sardet_paper120_nozoom_nocrop.md)。从 COCO 重新训练必须
使用新目录，不能覆盖已完成基线：

```bash
OUT=$(mktemp -d /root/zxy/experiments/deim_s_sardet_paper120_reproduce_XXXXXX)
export RUN OUT
python - <<'PY'
import os,pathlib,yaml
p=pathlib.Path(os.environ['OUT'])
c=yaml.safe_load((pathlib.Path(os.environ['RUN'])/'resolved_config.yml').read_text())
c.update(output_dir=str(p),summary_dir=str(p/'summary'))
(p/'resolved_config.yml').write_text(yaml.safe_dump(c,sort_keys=False))
PY
git rev-parse HEAD > "$OUT/git_revision.txt"
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -t /root/zxy/pretrain/deim/deim_dfine_hgnetv2_s_coco_120e.pth \
  --seed 0 --use-amp --output-dir "$OUT" \
  > "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
```

恢复只针对已停止的独立运行，明确设置 OUT，不对上述完成的 baseline 执行：

```bash
OUT=/root/zxy/experiments/替换为需要恢复的独立运行
LOG="$OUT/resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S).console.log"
ulimit -Sn 65535
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
  -c "$OUT/resolved_config.yml" -r "$OUT/last.pth" --seed 0 --use-amp --output-dir "$OUT" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$OUT/launcher.pid"
disown "$PID"
tail -f "$LOG"
```

当前 solver 的 last.pth 仅在 epoch<118 更新，正常最后为 epoch 117，不是最终权重。
最佳检查点仍按实测 AP 选择。原 baseline 的日志、指标与 TensorBoard：

```bash
OUT="$RUN"
tail -n 8 "$OUT/console.log"
tail -n 3 "$OUT/log.txt"
tensorboard --logdir "$OUT/summary" --host 0.0.0.0 --port 6006
```
