#!/usr/bin/env python3
"""Zero-training checks for two score-side components: is there evidence left, and what are the false positives?

export: runs one checkpoint on test and saves, for all queries of the evaluated decoder layer, the box, final
    and raw class scores, the LQE correction and statistics of the FDR edge distributions (one process, GPU or CPU).
analyze (CPU):
1. Evidence. Accurate low-score boxes of unseen targets are compared with background queries, with duplicates of
   located targets and with confident false positives. Separation is AUC; a fixed-form rescoring gives an upper
   bound on AP50. It is measured on test for diagnosis only and must not set any parameter.
2. False positives (score>=0.3, IoU<0.1 to every GT) are grouped into distinct objects with the crop matcher and
   shown once each with an id. Ids judged to be unlabelled aircraft can be passed to --unlabelled to see the
   AP50 with those boxes removed from the false positives. Annotations are never changed.

Writes OUTPUT/{report.md,summary.json,false_positive_groups.csv,false_positive_groups.png}.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_confusion_cues import CROP_WINDOW, auc, crop, to_8bit
from tools.analyze_sar_characteristics import DUP_SIDE, crop_vector, grey, grown, similar
from tools.compare_rem_102e import write_csv
from tools.diagnose_detection_errors import SCORE, SEEN_THRESHOLD, evaluate, fmt, ignore_others, seen_scores
from tools.sar_analysis_core import iou_matrix, load_json, save_json

BACKGROUND_IOU = .1
TOPK = 4  # the same top-k as the decoder's LQE
EVIDENCE = ('raw_score', 'lqe_delta', 'concentration', 'concentration_min', 'concentration_std', 'entropy')
RESCORED = ('concentration', 'concentration_min', 'entropy')
WEIGHTS = (.25, .5, 1., 2., 4.)
GROUP_THRESHOLD = .9
SHEET_GROUPS = 64
SHEET_COLUMNS = 8


def edge_statistics(corners, topk=TOPK):
    """Per-query statistics of the four FDR edge distributions; corners is B x L x 4(reg_max+1)."""
    import torch

    batch, queries, _ = corners.shape
    prob = corners.reshape(batch, queries, 4, -1).softmax(-1)
    concentration = prob.topk(topk, dim=-1).values.mean(-1)
    entropy = -(prob * prob.clamp_min(1e-12).log()).sum(-1) / torch.log(torch.tensor(float(prob.shape[-1])))
    return dict(concentration=concentration.mean(-1), concentration_min=concentration.min(-1).values,
                concentration_std=concentration.std(-1, unbiased=False), entropy=entropy.mean(-1))


def query_rows(raw, calibrated, corners, boxes, sizes):
    """Arrays for one batch: top class of each query, its scores, edge statistics and xywh box in source pixels."""
    import torch

    scores, labels = calibrated.sigmoid().max(-1)
    row = dict(score=scores, label=labels, raw_score=raw.sigmoid().gather(-1, labels[..., None])[..., 0],
               lqe_delta=(calibrated - raw).gather(-1, labels[..., None])[..., 0], **edge_statistics(corners))
    scale = sizes[:, None, :].repeat(1, boxes.shape[1], 1).to(boxes.dtype)
    centre, extent = boxes[..., :2] * scale, boxes[..., 2:] * scale
    row['box'] = torch.cat((centre - extent / 2, extent), dim=-1)
    return {k: v.detach().cpu().numpy() for k, v in row.items()}


def export(config, checkpoint, ann_file, output, device='cuda:0', batch_size=4, threads=16):
    import torch
    from torch.utils.data import DataLoader

    from engine.core import YAMLConfig

    torch.set_num_threads(threads)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = YAMLConfig(str(config))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'] = str(ann_file)
    model = cfg.model.to(device).eval()
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state['ema']['module'] if state.get('ema') else state['model'], strict=True)
    base = cfg.val_dataloader
    loader = DataLoader(base.dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=base.collate_fn)
    decoder = model.decoder.decoder
    captured = {}

    def hook(_, args, result):
        captured.update(raw=args[0], corners=args[1], calibrated=result)

    handle = decoder.lqe_layers[decoder.eval_idx].register_forward_hook(hook)
    rows, image_ids = defaultdict(list), []
    with torch.inference_mode():
        for step, (images, targets) in enumerate(loader):
            outputs = model(images.to(device))
            sizes = torch.stack([t['orig_size'] for t in targets]).to(device)
            batch = query_rows(captured['raw'], captured['calibrated'], captured['corners'], outputs['pred_boxes'], sizes)
            for key, value in batch.items():
                rows[key].append(value.reshape(-1, *value.shape[2:]))
            ids = [int(t['image_id'].item()) for t in targets]
            image_ids.append(np.repeat(ids, batch['score'].shape[1]))
            if step % 20 == 0:
                print(f'batch {step + 1}/{len(loader)}', flush=True)
    handle.remove()
    np.savez_compressed(output, image_id=np.concatenate(image_ids), checkpoint=str(checkpoint),
                        **{k: np.concatenate(v) for k, v in rows.items()})
    print(output, flush=True)


def group_queries(dataset, rows, seen):
    """Boolean masks of queries: best box of each target by outcome, duplicates, partial boxes and background."""
    by_image = defaultdict(list)
    for index, image_id in enumerate(rows['image_id']):
        by_image[int(image_id)].append(index)
    gts = defaultdict(list)
    for a in dataset['annotations']:
        gts[a['image_id']].append(a)
    masks = {name: np.zeros(len(rows['score']), bool) for name in
             ('seen_located', 'seen_low', 'unseen_located', 'unseen_low', 'duplicate', 'partial', 'background', 'false_positive')}
    outrank = []
    for image_id, indices in by_image.items():
        indices = np.array(indices)
        scores, boxes = rows['score'][indices], rows['box'][indices]
        annotations = gts[image_id]
        if not annotations:
            masks['background'][indices[scores < SCORE]] = True
            masks['false_positive'][indices[scores >= SCORE]] = True
            continue
        overlap = iou_matrix(boxes, [a['bbox'] for a in annotations])
        best = overlap.max(axis=1)
        chosen = set()
        background = indices[best < BACKGROUND_IOU]
        for g, a in enumerate(annotations):
            hits = np.nonzero(overlap[:, g] >= .5)[0]
            if not hits.size:
                continue
            top = hits[np.argmax(scores[hits])]
            chosen.add(int(top))
            kind = 'seen' if seen[a['id']] else 'unseen'
            located = scores[top] >= SCORE
            masks[f"{kind}_{'located' if located else 'low'}"][indices[top]] = True
            if kind == 'unseen' and not located:
                outrank.append(int((rows['score'][background] > scores[top]).sum()))
        covered = best >= .5
        for position in np.nonzero(covered)[0]:
            if int(position) not in chosen:
                masks['duplicate'][indices[position]] = True
        masks['partial'][indices[(best >= BACKGROUND_IOU) & (best < .5)]] = True
        low = best < BACKGROUND_IOU
        masks['background'][indices[low & (scores < SCORE)]] = True
        masks['false_positive'][indices[low & (scores >= SCORE)]] = True
    return masks, outrank


def separation(rows, masks):
    pairs = (('unseen_low', 'background'), ('unseen_low', 'duplicate'), ('unseen_low', 'false_positive'),
             ('unseen_located', 'false_positive'), ('unseen_low', 'seen_located'))
    table = []
    for a, b in pairs:
        row = dict(positive=a, negative=b, positive_n=int(masks[a].sum()), negative_n=int(masks[b].sum()))
        for key in EVIDENCE:
            row[key] = auc(rows[key][masks[a]], rows[key][masks[b]])
        table.append(row)
    return table


def medians(rows, masks):
    return {name: dict(n=int(mask.sum()), score=float(np.median(rows['score'][mask])) if mask.any() else None,
                       **{k: float(np.median(rows[k][mask])) if mask.any() else None for k in EVIDENCE})
            for name, mask in masks.items()}


def as_predictions(rows, scores, keep=None):
    keep = np.ones(len(scores), bool) if keep is None else keep
    return [dict(image_id=int(i), category_id=int(c), bbox=[float(v) for v in b], score=float(s))
            for i, c, b, s in zip(rows['image_id'][keep], rows['label'][keep], rows['box'][keep], scores[keep])]


def rescored(rows, key, weight):
    """Fixed-form upper bound: logit(score) + weight * z(evidence), with larger evidence meaning more certain."""
    values = rows[key] if key != 'entropy' else -rows[key]
    z = (values - values.mean()) / max(values.std(), 1e-9)
    score = np.clip(rows['score'], 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(score / (1 - score)) + weight * z)))


def rescoring_table(dataset, rows, groups, output):
    views = {name: ignore_others(dataset, keep) for name, keep in groups.items()}

    def line(scores):
        return {name: evaluate(view, as_predictions(rows, scores), output)['AP50'] for name, view in views.items()}

    base = line(rows['score'])
    table = [dict(evidence=key, weight=weight, **line(rescored(rows, key, weight))) for key in RESCORED for weight in WEIGHTS]
    return base, table


def box_contrast(values, box):
    x0, y0, x1, y1 = grown(box, 0, values.shape[1], values.shape[0])
    patch = values[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    flat = np.sort(patch.ravel())
    top = flat[-max(1, len(flat) // 10):].mean()
    rx0, ry0, rx1, ry1 = grown(box, .5, values.shape[1], values.shape[0])
    ring = values[ry0:ry1, rx0:rx1].copy()
    ring[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0] = np.nan
    level = np.nanmedian(ring) if np.isfinite(ring).any() else 0.
    return float(20 * np.log10((top + 1) / (level + 1)))


def clipped(box, width, height):
    x0, y0 = max(box[0], 0.), max(box[1], 0.)
    x1, y1 = min(box[0] + box[2], width), min(box[1] + box[3], height)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)] if x1 - x0 >= 2 and y1 - y0 >= 2 else None


def false_positive_groups(dataset, rows, mask, data_root):
    """One record per distinct object among the false positives, largest group first; ids start at 1."""
    names = {c['id']: c['name'] for c in dataset['categories']}
    images = {im['id']: im for im in dataset['images']}
    records, vectors = [], []
    for index in np.nonzero(mask)[0]:
        im = images[int(rows['image_id'][index])]
        box = clipped(rows['box'][index], im['width'], im['height'])
        if box is None:
            continue
        values, _ = grey(data_root / im['file_name'])
        records.append(dict(index=int(index), image_id=im['id'], bbox=box, score=float(rows['score'][index]),
                            category=names[int(rows['label'][index])], size=max(box[2:]), contrast=box_contrast(values, box)))
        vectors.append(crop_vector(values, box))
    if not records:
        return [], {}
    vectors = np.array(vectors, np.float32).reshape(-1, DUP_SIDE * DUP_SIDE)
    sizes = np.array([r['size'] for r in records], float)
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, score in similar(vectors, vectors, sizes, sizes, same=True):
        if score >= GROUP_THRESHOLD:
            parent[find(i)] = find(j)
    members = defaultdict(list)
    for i, record in enumerate(records):
        members[find(i)].append(record)
    ordered = sorted(members.values(), key=lambda m: (-len(m), -max(r['score'] for r in m)))
    groups, assignment = [], {}
    for number, group in enumerate(ordered, 1):
        best = max(group, key=lambda r: r['score'])
        groups.append(dict(group=number, boxes=len(group), images=len({r['image_id'] for r in group}), top_score=best['score'],
                           category=Counter(r['category'] for r in group).most_common(1)[0][0], size=float(np.median([r['size'] for r in group])),
                           contrast=float(np.median([r['contrast'] for r in group])), image_id=best['image_id'], bbox=best['bbox']))
        assignment.update({r['index']: number for r in group})
    return groups, assignment


def group_sheet(path, groups, dataset, data_root):
    files = {im['id']: im['file_name'] for im in dataset['images']}
    shown = groups[:SHEET_GROUPS]
    rows = -(-len(shown) // SHEET_COLUMNS)
    sheet = Image.new('RGB', (SHEET_COLUMNS * CROP_WINDOW, 14 + rows * CROP_WINDOW), (0, 0, 0))
    ImageDraw.Draw(sheet).text((4, 1), f'false positive objects, largest first, showing {len(shown)} of {len(groups)}: G<id> x<boxes> <score> <class>', fill=(255, 255, 255))
    for i, g in enumerate(shown):
        image = to_8bit(Image.open(data_root / files[g['image_id']]))
        tile = crop(image, g['bbox'], f"G{g['group']} x{g['boxes']} {g['top_score']:.2f} {g['category'][:7]}", (255, 200, 0))
        sheet.paste(tile, ((i % SHEET_COLUMNS) * CROP_WINDOW, 14 + (i // SHEET_COLUMNS) * CROP_WINDOW))
    sheet.save(path)


def analyze(evidence_path, data_root, output, reference='trainval', split='test', unlabelled=()):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    loaded = np.load(evidence_path)
    rows = {k: loaded[k] for k in loaded.files if k != 'checkpoint'}
    dataset = load_json(data_root / f'COCO/{split}.json')
    scores = seen_scores(dataset, load_json(data_root / f'COCO/{reference}.json'), data_root)
    seen = {k: v >= SEEN_THRESHOLD for k, v in scores.items()}
    masks, outrank = group_queries(dataset, rows, seen)
    views = dict(all=set(seen), seen={k for k, v in seen.items() if v}, unseen={k for k, v in seen.items() if not v})
    base, table = rescoring_table(dataset, rows, views, output)
    groups, assignment = false_positive_groups(dataset, rows, masks['false_positive'], data_root)
    positives = sum(g['boxes'] for g in groups)
    removal = {}
    if unlabelled:
        drop = np.array([assignment.get(i) in set(unlabelled) for i in range(len(rows['score']))])
        removal = dict(groups=list(unlabelled), boxes=int(drop.sum()), AP50=evaluate(dataset, as_predictions(rows, rows['score'], ~drop), output)['AP50'])
    gt_contrast = []
    images = {im['id']: im for im in dataset['images']}
    by_image = defaultdict(list)
    for a in dataset['annotations']:
        by_image[a['image_id']].append(a)
    for image_id, annotations in by_image.items():
        values, _ = grey(data_root / images[image_id]['file_name'])
        gt_contrast += [box_contrast(values, a['bbox']) for a in annotations]
    summary = dict(split=split, reference=reference, checkpoint=str(loaded['checkpoint']), medians=medians(rows, masks), separation=separation(rows, masks),
                   background_outranking_unseen_low=dict(p10=float(np.quantile(outrank, .1)), p50=float(np.median(outrank)), p90=float(np.quantile(outrank, .9))) if outrank else None,
                   rescoring=dict(base=base, table=table), false_positives=dict(boxes=positives, distinct_objects=len(groups),
                   top5_share=sum(g['boxes'] for g in groups[:5]) / positives if positives else None,
                   contrast_p50=float(np.median([g['contrast'] for g in groups])) if groups else None, gt_contrast_p50=float(np.median(gt_contrast)),
                   by_class=dict(Counter(g['category'] for g in groups)), groups=groups), unlabelled=removal,
                   definitions=dict(unseen_low='highest-scoring query with IoU>=0.5 to an unseen GT, score<0.3', duplicate='IoU>=0.5 to a GT, not the highest-scoring query of that GT',
                                    background=f'IoU<{BACKGROUND_IOU} to every GT, score<{SCORE}', false_positive=f'IoU<{BACKGROUND_IOU} to every GT, score>={SCORE}',
                                    concentration='mean over the four edges of the mean top-4 FDR probability, as in LQE', AUC='P(first group larger than second); 0.5 = no separation'))
    save_json(output / 'summary.json', summary)
    write_csv(output / 'false_positive_groups.csv', [dict(g, bbox=' '.join(f'{v:.1f}' for v in g['bbox'])) for g in groups])
    if groups:
        group_sheet(output / 'false_positive_groups.png', groups, dataset, data_root)
    (output / 'report.md').write_text(render(summary), encoding='utf-8')
    print(output / 'report.md', flush=True)
    return summary


def render(s):
    lines = [f"# 分数证据与误检（{s['split']}）", '', f"权重：`{s['checkpoint']}`", '', '## 1. 各组查询的中位数', '',
             '| 组 | 查询数 | score | ' + ' | '.join(EVIDENCE) + ' |', '|---|---:|---:|' + '---:|' * len(EVIDENCE)]
    for name, m in s['medians'].items():
        lines.append(f"| {name} | {m['n']} | {fmt(m['score'], 4)} | " + ' | '.join(fmt(m[k], 3) for k in EVIDENCE) + ' |')
    lines += ['', '## 2. 可分性（AUC）', '', '切片不独立，同一架飞机重复出现，AUC 只作方向参考。', '',
              '| 正组 | 负组 | ' + ' | '.join(EVIDENCE) + ' |', '|---|---|' + '---:|' * len(EVIDENCE)]
    for r in s['separation']:
        lines.append(f"| {r['positive']}（{r['positive_n']}） | {r['negative']}（{r['negative_n']}） | " + ' | '.join(fmt(r[k], 3) for k in EVIDENCE) + ' |')
    o = s['background_outranking_unseen_low']
    if o:
        lines += ['', f"同一张图里分数高于这些低分目标的背景查询数：p10 {o['p10']:.0f}，p50 {o['p50']:.0f}，p90 {o['p90']:.0f}。"]
    b = s['rescoring']['base']
    lines += ['', '## 3. 固定形式重打分的 AP50 上限（只用于诊断，不得据此设定参数）', '',
              f"基准（每个查询只保留最高类别）：全部 {fmt(b['all'])}，见过 {fmt(b['seen'])}，没见过 {fmt(b['unseen'])}。", '',
              '| 证据 | 权重 | 全部 | 见过 | 没见过 |', '|---|---:|---:|---:|---:|']
    lines += [f"| {r['evidence']} | {r['weight']:g} | {fmt(r['all'])} | {fmt(r['seen'])} | {fmt(r['unseen'])} |" for r in s['rescoring']['table']]
    lines += ['', '组件 1 继续的条件：unseen_low 对 background、对 duplicate 都明显可分（不只对 background），且 AP50 上限至少高 1。', '']
    f = s['false_positives']
    lines += ['## 4. 误检分组', '', f"{f['boxes']} 个框来自 {f['distinct_objects']} 个不同的物体，前 5 个占 {fmt(f['top5_share'], 3)}；"
              f"对比度中位数 {fmt(f['contrast_p50'], 1)} dB（GT {fmt(f['gt_contrast_p50'], 1)} dB）。", '',
              '`false_positive_groups.png` 每个物体一格，标注 `G编号 x框数 分数 类别`，请标出像漏标飞机的编号。', '',
              '| 编号 | 框数 | 图数 | 最高分 | 类别 | 尺寸 | 对比度 dB |', '|---:|---:|---:|---:|---|---:|---:|']
    lines += [f"| G{g['group']} | {g['boxes']} | {g['images']} | {fmt(g['top_score'])} | {g['category']} | {fmt(g['size'], 0)} | {fmt(g['contrast'], 1)} |" for g in f['groups'][:30]]
    if s['unlabelled']:
        u = s['unlabelled']
        lines += ['', f"把 {u['groups']} 视为漏标、移出误检（{u['boxes']} 个框）后 AP50：{fmt(u['AP50'])}（基准 {fmt(b['all'])}）。"]
    lines += ['', '## 定义', ''] + [f'- {k}：{v}' for k, v in s['definitions'].items()]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='action', required=True)
    first = sub.add_parser('export')
    first.add_argument('-c', '--config', type=Path, required=True)
    first.add_argument('-r', '--checkpoint', type=Path, required=True)
    first.add_argument('--ann-file', type=Path, required=True)
    first.add_argument('--output', type=Path, required=True, help='new .npz file')
    first.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')
    first.add_argument('--batch-size', type=int, default=4)
    first.add_argument('--threads', type=int, default=16)
    second = sub.add_parser('analyze')
    second.add_argument('--evidence', type=Path, required=True)
    second.add_argument('--data-root', type=Path, required=True)
    second.add_argument('--output', type=Path, required=True, help='new output directory (must not exist)')
    second.add_argument('--reference', default='trainval', help='split the run was trained on')
    second.add_argument('--split', default='test')
    second.add_argument('--unlabelled', type=int, nargs='*', default=[], help='false-positive group ids judged to be unlabelled aircraft')
    args = parser.parse_args()
    if args.action == 'export':
        export(args.config, args.checkpoint, args.ann_file, args.output, args.device, args.batch_size, args.threads)
    else:
        analyze(args.evidence, args.data_root, args.output, args.reference, args.split, args.unlabelled)


if __name__ == '__main__':
    main()
