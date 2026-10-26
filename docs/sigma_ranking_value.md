# 第 1 步：σ 在分数之外是否还有排序价值（交叉拟合 AP 实验）

## 背景：已有结果与唯一未决问题

第 0 步（`docs/fdr_variance_step0.md`）在 SARDet-100K 六类基线（AP 43.18）上得到：

| 项 | 结果 |
| --- | --- |
| 分布重建自检 | 2.4e-07，分布与 `pred_boxes` 同源 |
| `corr(σ, |边误差|)` 四边中位数 | 0.430 |
| 四边 σ 各向异性 | `max/min = 1.0498`，L/R=0.990，T/B=0.990 |
| 校准单调性 | 四边全单调，q1→q5 误差 0.10→0.45 |
| `corr(mean_σ, 1−IoU)` | +0.583 |
| `corr(mean_σ, score)` | **−0.811** |

结论分两半：

- **各向异性假设被证伪。** 四边 σ 差 1%，在 1 个标准误内不可区分，实际误差也各向同性
  （L 0.244 / T 0.236 / R 0.258 / B 0.240）。「SAR 阴影侧边缘更不确定」这个物理叙事
  在本检查点、本数据上不成立，基于它的模块不再考虑。
- **σ 与误差有稳定的单调关系**，但 σ 与分数高度共线（−0.811）。

因此唯一未决的问题是：**σ 里有多少信息是分数没有的？**

## 为什么不能只算相关系数

已知 `corr(σ, 1−IoU)=0.583` 与 `corr(σ, score)=−0.811`。σ 对 IoU 的净贡献取决于
`ρ = corr(score, IoU)`，而 ρ 尚未测量。偏相关会给出从冗余到强独立的全谱结果，
但**偏相关仍然不是论文需要的量**。论文需要的是 AP 增益。

所以本实验直接用 AP 回答，不经过相关系数中转。

## 实验设计：交叉拟合的分数重排

`tools/localization_ceiling.py:129-130` 的 `AP_IQS` 是把每个检测的分数乘以它与最佳
GT 的 IoU。+8.02 就是这个 oracle 的上限。本实验问的是：**用 σ 拟合出的预测 IoU
去乘分数，能拿到其中多少？**

对每个检测计算：

- `s`：现有分数
- `σ`：四边分布标准差取均值（主），另报告四边最大值
- `iou*`：与该检测同类 GT 的最大 IoU（无同类 GT 时为 0），与 `AP_IQS` 同定义

然后在验证集上按图像做**两折交叉拟合**（拟合与评估严格不重叠，避免自我实现的 AP）：

| 变体 | 排序分数 | 作用 |
| --- | --- | --- |
| `AP` | `s` | 参考，必须复现 43.18 |
| `AP_IQS` | `s × iou*` | oracle 上限，应复现 51.20 |
| `AP_const` | `s × mean(iou*)` | 对照：常数重标定，等价于 AP |
| `AP_score` | `s × f(σ)` 换为 `s × g(s)` | 只用分数自身校准 IoU，**这是基线** |
| `AP_sigma` | `s × g(σ)` | 只用 σ |
| `AP_both` | `s × h(s, σ)` | 两者联合 |

判定的关键是 **`AP_both` 相对 `AP_score` 的增量**，不是 `AP_sigma` 本身。
若 `AP_both ≈ AP_score`，说明 σ 的信息已经全部在分数里，方向关闭。

`f/g/h` 用非参数分位分箱拟合，不引入可训练参数，也不需要 sklearn。

## 第 1 步：重新导出，覆盖完整查询集

第 0 步的导出用了 `--score-thr 0.05 --topk 100`，截断了排序尾部，而 AP 依赖完整排名。
本次必须用与 `ceiling.txt` 一致的检测集。

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
RUN=<替换为实际基线运行目录>
DIAG="$RUN/diagnosis"
export CUDA_VISIBLE_DEVICES=0
python tools/export_corner_stats.py \
  -c "$RUN/resolved_config.yml" \
  -r "$RUN/best_stg1.pth" \
  -o "$DIAG/corners_full.npz" \
  --score-thr 0.001 --topk 300
```

4096 张图 × 至多 300 查询 ≈ 120 万条记录，npz 约 100–200 MB，单卡十几分钟。

保留第 0 步的 `corners.npz` 不动，两份并存。

## 第 2 步：新增分析脚本

新建 `tools/sigma_ranking_value.py`：

```python
#!/usr/bin/env python3
"""Measure whether FDR edge spread adds ranking value beyond the detection score.

Motivation: on the SARDet subset, corr(mean_sigma, 1-IoU) = 0.583 but
corr(mean_sigma, score) = -0.811. Correlations cannot separate "sigma adds
information" from "sigma restates the score" when the two are this collinear,
and neither answer is what a paper needs: the quantity of interest is AP.

This script therefore asks the AP question directly. For every detection it
computes the max IoU against same-class ground truth (the same quantity
AP_IQS uses), then cross-fits a post-hoc quality map from score and/or sigma
to that IoU on one half of the validation images and evaluates AP on the other.
Fit and evaluation never share an image.

Variants:
  AP        reference, must reproduce the ceiling.txt AP
  AP_IQS    s * iou*, the oracle upper bound
  AP_const  s * mean(iou*), a control that must land on AP
  AP_score  s * g(score)          score-only calibration, the baseline to beat
  AP_sigma  s * g(sigma)          sigma alone
  AP_both   s * h(score, sigma)   the test: does this beat AP_score

Usage:
python tools/sigma_ranking_value.py --npz corners_full.npz --gt val.json \
    --output ranking.txt
"""

import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

EPS = 1e-6


def iou_1toN(box, boxes):
    if len(boxes) == 0:
        return np.zeros(0)
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    bx1, by1 = box[0], box[1]
    bx2, by2 = box[0] + box[2], box[1] + box[3]
    gx1, gy1, gx2, gy2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(bx2, gx2) - np.maximum(bx1, gx1), 0, None)
    ih = np.clip(np.minimum(by2, gy2) - np.maximum(by1, gy1), 0, None)
    inter = iw * ih
    union = box[2] * box[3] + b[:, 2] * b[:, 3] - inter
    out = np.zeros(len(b))
    ok = union > 0
    out[ok] = inter[ok] / union[ok]
    return out


def quantile_map(fit_values, fit_targets, n_bins=8):
    """Piecewise-constant map from a scalar to its conditional mean target."""
    edges = np.quantile(fit_values, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return lambda v: np.full_like(np.asarray(v, np.float64), fit_targets.mean())
    idx = np.clip(np.searchsorted(edges, fit_values, side='right') - 1, 0, len(edges) - 2)
    table = np.array([fit_targets[idx == b].mean() if np.any(idx == b)
                      else fit_targets.mean() for b in range(len(edges) - 1)])

    def apply(v):
        j = np.clip(np.searchsorted(edges, v, side='right') - 1, 0, len(edges) - 2)
        return table[j]

    return apply


def joint_map(fit_s, fit_g, fit_targets, n_bins=6):
    """2D piecewise-constant map, falling back to marginals where a cell is sparse."""
    f_s = quantile_map(fit_s, fit_targets, n_bins)
    f_g = quantile_map(fit_g, fit_targets, n_bins)
    es = np.unique(np.quantile(fit_s, np.linspace(0, 1, n_bins + 1)))
    eg = np.unique(np.quantile(fit_g, np.linspace(0, 1, n_bins + 1)))
    if len(es) < 3 or len(eg) < 3:
        return f_s

    def cell(v, edges):
        return np.clip(np.searchsorted(edges, v, side='right') - 1, 0, len(edges) - 2)

    i = cell(fit_s, es)
    j = cell(fit_g, eg)
    key = i * len(eg) + j
    n_cells = len(es) * len(eg)
    table = np.full(n_cells, np.nan)
    for k in np.unique(key):
        sel = key == k
        if sel.sum() >= 50:
            table[k] = fit_targets[sel].mean()

    def apply(s, g):
        i2 = cell(s, es)
        j2 = cell(g, eg)
        k2 = i2 * len(eg) + j2
        out = table[k2].copy()
        fallback = ~np.isfinite(out)
        if fallback.any():
            # Sparse cells fall back to the average of both marginals, which is
            # smoother than either alone and never extrapolates wildly.
            out[fallback] = 0.5 * (f_s(s[fallback]) + f_g(g[fallback]))
        return out

    return apply


def subset_coco(gt_json, image_ids):
    """A COCO gt holding only `image_ids`.

    Evaluating fold-k detections against the full gt would count every image in
    the other fold as a missed detection, since COCOeval treats gt images with
    no detections as pure false negatives. AP would collapse for a reason that
    has nothing to do with the quality map.
    """
    keep = set(int(i) for i in image_ids)
    sub = {
        'images': [im for im in gt_json['images'] if int(im['id']) in keep],
        'annotations': [a for a in gt_json['annotations'] if int(a['image_id']) in keep],
        'categories': gt_json['categories'],
    }
    coco = COCO()
    coco.dataset = sub
    coco.createIndex()
    return coco


def evaluate(coco_gt, dets, name):
    if not dets:
        print(f'{name}: no detections')
        return None
    coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.evaluate()
    ev.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        ev.summarize()
    return ev.stats


def collect(z, gt, score_thr):
    """Turn the exported statistics into per-detection arrays with iou*."""
    images = {im['id']: im for im in gt['images']}

    keep = z['score'] >= score_thr
    image_id = z['image_id'][keep]
    label = z['label'][keep]
    score = z['score'][keep].astype(np.float64)
    box = z['box'][keep].astype(np.float64)
    sigma_mean = z['std'][keep].astype(np.float64).mean(-1)
    sigma_max = z['std'][keep].astype(np.float64).max(-1)

    best_iou = np.zeros(len(score))
    boxes_px = np.zeros((len(score), 4))
    gts_by_img = defaultdict(list)
    for ann in gt['annotations']:
        if not ann.get('iscrowd', 0):
            gts_by_img[ann['image_id']].append(ann)
    per_img = defaultdict(list)
    for n, iid in enumerate(image_id):
        per_img[int(iid)].append(n)

    for iid, rows in per_img.items():
        im = images.get(iid)
        if im is None:
            continue
        W, H = im['width'], im['height']
        cand = gts_by_img.get(iid, [])
        for n in rows:
            b = box[n]
            boxes_px[n] = [(b[0] - b[2] / 2) * W, (b[1] - b[3] / 2) * H, b[2] * W, b[3] * H]
            same = [a for a in cand if a['category_id'] == label[n]]
            if same:
                ious = iou_1toN(boxes_px[n], [a['bbox'] for a in same])
                best_iou[n] = float(ious.max()) if len(ious) else 0.0

    return dict(image_id=image_id, label=label, score=score, box=boxes_px,
                sigma_mean=sigma_mean, sigma_max=sigma_max, best_iou=best_iou)


def build_dets(d, quality):
    out = []
    for n in range(len(d['score'])):
        x, y, w, h = d['box'][n]
        out.append({'image_id': int(d['image_id'][n]),
                    'category_id': int(d['label'][n]),
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(max(d['score'][n] * quality[n], 0.0))})
    return out


def corr(a, b):
    if len(a) < 30 or a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--score-thr', type=float, default=0.001)
    ap.add_argument('--folds', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output')
    args = ap.parse_args()

    with np.load(args.npz) as archive:
        z = {k: archive[k] for k in archive.files}
    gt_json = json.loads(Path(args.gt).read_text())
    gt = COCO(args.gt)

    d = collect(z, gt_json, args.score_thr)
    n = len(d['score'])
    lines = []

    def emit(text=''):
        print(text)
        lines.append(text)

    emit(f'detections: {n}  (score >= {args.score_thr})')
    emit(f'corr(score, iou*)      = {corr(d["score"], d["best_iou"]):+.4f}')
    emit(f'corr(sigma_mean, iou*) = {corr(d["sigma_mean"], d["best_iou"]):+.4f}')
    emit(f'corr(sigma_max,  iou*) = {corr(d["sigma_max"], d["best_iou"]):+.4f}')
    emit(f'corr(sigma_mean, score)= {corr(d["sigma_mean"], d["score"]):+.4f}')
    emit(f'corr(sigma_mean, size) = {corr(d["sigma_mean"], np.sqrt(d["box"][:, 2] * d["box"][:, 3])):+.4f}')

    rng = np.random.default_rng(args.seed)
    imgs = np.unique(d['image_id'])
    rng.shuffle(imgs)
    fold_of_img = {int(i): k % args.folds for k, i in enumerate(imgs)}
    fold = np.array([fold_of_img[int(i)] for i in d['image_id']])

    log_sigma = np.log(d['sigma_mean'] + EPS)
    names = ['AP', 'AP_IQS', 'AP_const', 'AP_score', 'AP_sigma', 'AP_both']
    acc = {k: [] for k in names}

    for k in range(args.folds):
        fit = fold != k
        ev = fold == k
        if ev.sum() < 100 or fit.sum() < 100:
            continue

        # Evaluate against this fold's GT only, so the other fold's images do
        # not count as missed detections.
        fold_gt = subset_coco(gt_json, d['image_id'][ev])

        g_s = quantile_map(d['score'][fit], d['best_iou'][fit])
        g_g = quantile_map(log_sigma[fit], d['best_iou'][fit])
        h = joint_map(d['score'][fit], log_sigma[fit], d['best_iou'][fit])

        variants = {
            'AP': np.ones(ev.sum()),
            'AP_IQS': d['best_iou'][ev],
            'AP_const': np.full(ev.sum(), d['best_iou'][fit].mean()),
            'AP_score': g_s(d['score'][ev]),
            'AP_sigma': g_g(log_sigma[ev]),
            'AP_both': h(d['score'][ev], log_sigma[ev]),
        }

        sub = {kk: vv[ev] for kk, vv in d.items() if isinstance(vv, np.ndarray)}
        for name, q in variants.items():
            s = evaluate(fold_gt, build_dets(sub, np.clip(q, 0.0, 1.0)), f'{name} fold{k}')
            acc[name].append(s[0] * 100 if s is not None else float('nan'))

    emit('')
    emit(f'cross-fitted AP over {args.folds} image folds:')
    emit(f'{"variant":10s} {"mean AP":>8s} {"per fold":>26s}  meaning')
    meaning = {
        'AP': 'reference',
        'AP_IQS': 'oracle upper bound',
        'AP_const': 'control, should equal AP',
        'AP_score': 'score-only calibration = baseline to beat',
        'AP_sigma': 'sigma alone',
        'AP_both': 'score + sigma = the test',
    }
    for name in names:
        v = [x for x in acc[name] if np.isfinite(x)]
        if not v:
            emit(f'{name:10s} {"n/a":>8s}')
            continue
        emit(f'{name:10s} {np.mean(v):8.2f} {str([round(x, 2) for x in v]):>26s}  {meaning[name]}')

    if acc['AP_score'] and acc['AP_both']:
        a = np.mean([x for x in acc['AP_score'] if np.isfinite(x)] or [np.nan])
        b = np.mean([x for x in acc['AP_both'] if np.isfinite(x)] or [np.nan])
        emit('')
        emit(f'delta(AP_both - AP_score) = {b - a:+.2f} AP')
        emit('This delta is the only quantity that decides whether sigma carries')
        emit('ranking information the score does not already have.')

    if args.output:
        Path(args.output).write_text('\n'.join(lines) + '\n')
        print(f'\nwritten {args.output}')


if __name__ == '__main__':
    main()
```

## 第 3 步：执行

```bash
cd /root/zxy/deim
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate deim_zxy
RUN=<替换为实际基线运行目录>
DIAG="$RUN/diagnosis"
python tools/sigma_ranking_value.py \
  --npz "$DIAG/corners_full.npz" \
  --gt /root/zxy/experiments/data_subsets/sardet100k_screen_v1/val.json \
  --output "$DIAG/sigma_ranking.txt"
cat "$DIAG/sigma_ranking.txt"
```

纯 CPU。120 万条检测、两折、每折 6 个变体，约 12 次 COCOeval，预计十几分钟。

## 判读标准

先看两个锚点，它们证明流程正确：

1. `AP` 应复现 `ceiling.txt` 的 **43.18**（±0.3）。这里是交叉折均值，单折略有波动。
2. `AP_IQS` 应复现 **51.20**（±0.5）。
3. `AP_const` 应接近 `AP`。偏离超过 0.3 说明质量映射有 bug。

锚点全部对上路后，只看一个数：

| `delta(AP_both − AP_score)` | 结论 |
| --- | --- |
| ≥ +1.5 | σ 有独立排序价值。可做纯后处理模块，**不需要训练**，成本极低 |
| +0.5 ~ +1.5 | 弱信号。值得做但要配合定位改进，单独不够一篇 |
| −0.5 ~ +0.5 | **冗余，方向关闭**。σ 的信息已在分数中 |
| < −0.5 | σ 有害，拟合过拟合或映射错误，先查 bug |

附加参考：`AP_sigma` 若明显低于 `AP_score`，说明 σ 单独的信息量弱，
但这不是否定的充分条件，因为 σ 可能只在分数不确定的区段有效。
真正的判定仍是 `AP_both − AP_score`。

## 必须注意的局限

- **这是后处理上限，不是可训练方法的上限。** `g`/`h` 在 val 上交叉拟合，
  实际部署需要在 train 上训练一个映射头。两者量级接近但不相等，
  交叉拟合结果是乐观估计。
- **交叉拟合只能防止自我实现，不能防止 val 过拟合。** 两折、每折 2000+ 图，
  拟合自由度很低（分位分箱，最多 6×6 格），过拟合风险小但非零。
- **`AP_sigma` 与 `AP_both` 的分数仍是 `s × q`**，与 `AP_IQS` 同一形式，
  可比。但若将来改成加权和，需重新对齐口径。
- 第 0 步记录的 `valid_pairs_rejected_by_document_index_filter: 5546`
  只是一个诊断计数器：这些对被打上标记但仍计入 `retained`，
  最终样本数 6031 正确，不影响结论。

## 交付要求

回填到本文件：

- 三个锚点的实测值（`AP`、`AP_IQS`、`AP_const`）。
- 五个相关系数（score/iou*、sigma_mean/iou*、sigma_max/iou*、sigma_mean/score、sigma_mean/size）。
- 六变体的交叉折 AP 与每折数值。
- `delta(AP_both − AP_score)`。
- 按判读表给出的结论，并说明依据。
- 是否观察到 `delta` 在某一尺寸档或某一类别上集中（若脚本未输出，注明未测）。

未执行的部分写「未执行」及原因。

## 若通过的下一步（暂不实现）

若 `delta ≥ +1.5`：

1. 用 train 集训练一个轻量映射头（输入 score、σ、尺寸、类别，输出校准 IoU），
   替换交叉拟合中的 `h`。
2. 该头与检测器**解耦**，不改 D-FINE 结构，只改推理时的排序。
3. 消融：仅 `g(score)`、仅 `g(σ)`、`h(score,σ)`，报告 AP/AP75/APs。

若未通过：不再使用 σ。转向 `AP_TPFP` 的 **+4.66**（OGSOD 上仅 +1.05，
说明 SARDet 上前后景分离远未饱和，OGSOD 的否定结论不可迁移）。
