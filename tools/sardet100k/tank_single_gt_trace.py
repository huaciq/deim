"""Follow single GTs from the input to the final box: one TP, one Loc, one LowScore, one NoBox, all from one image.

Usage: python tools/sardet100k/tank_single_gt_trace.py RUN OUT [--image 0112914.png] [--device cuda:0]

OUT is the tank_feature_diagnosis directory. Writes OUT/trace/: one figure per sample, a comparison sheet, trace.json, population.json.
Taking the four from one image holds scene, imaging version and scale fixed, so differences belong to the targets.
Weights and code on disk are unchanged; decoder layers are reached by setting eval_idx at runtime. A query keeps its index
through the decoder, so one index is one candidate from the encoder to the output.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_gallery import ROOT, iou_matrix, match_image  # noqa: E402
from tank_feature_diagnosis import FONT, GT_C, INPUT, SCORE, SMALL, STRIDES, TANK, Hooks, colour, l2_map, load_model, load_val, panel, region_stats, sampled, to_input, xyxy  # noqa: E402

LEVEL_COLOURS = {8: (70, 190, 255), 16: (255, 200, 0), 32: (255, 80, 200)}
TRACK_COLOURS = ((255, 60, 200), (0, 230, 230))
FIELDS = ('input_side', 'enc_prob_max_in_gt', 'enc_best_iou_selected', 'q_final_score', 'q_final_iou')
STAGES = ('P3', 'P4', 'P5', 'F3', 'F4')
TOP = 300
CELL = 230
BIG_SPAN, BOX_SPAN = 120, 64


def strict_stage(r):
    if r['native_gw'] <= 0 or r['native_gh'] <= 0:
        return 'zero-area'
    if r['status'] == 'TP':
        return 'detected'
    if r['enc_selected_in_gt'] == 0 and r['enc_best_iou_selected'] < .1:
        return 'no selected query'
    if r['dec3_best_iou'] < .5:
        return 'no box IoU>=.5'
    return 'box ok, low score'


def choose(rows, image):
    """The GT closest to its own status median on the funnel fields, among GTs that fit the question asked of that status."""
    wanted = dict(A=('TP', 'detected'), B=('Loc', 'no box IoU>=.5'), C=('LowScore', 'box ok, low score'), D=('NoBox', 'no selected query'))
    ogsod = [r for r in rows if r['source'] == 'ogsod']
    chosen = {}
    for name, (status, stage) in wanted.items():
        group = [r for r in ogsod if r['status'] == status and strict_stage(r) == stage]
        centre = {f: float(np.median([r[f] for r in group])) for f in FIELDS}
        spread = {f: float(np.std([r[f] for r in group])) + 1e-6 for f in FIELDS}
        pool = [r for r in group if r['file_name'] == image]
        pool.sort(key=lambda r: (sum(((r[f] - centre[f]) / spread[f]) ** 2 for f in FIELDS), r['gt_id']))
        chosen[name] = dict(row=pool[0], status=status, stage=stage, group_size=len(group), group_median={k: round(v, 3) for k, v in centre.items()}, candidates=len(pool))
    return chosen


class Tracer:
    def __init__(self, model):
        self.hooks = Hooks(model)
        self.model = model
        self.raw = {}
        for k, head in enumerate(model.decoder.dec_score_head):
            head.register_forward_hook(self.keep(k))

    def keep(self, k):
        def hook(_, __, output):
            self.raw[k] = output.detach()
        return hook

    def run(self, x):
        """Everything per decoder layer for all queries. Boxes are normalised cxcywh until converted by the caller."""
        import torch
        stages, enc, _, _ = self.hooks.run(x)  # Fills the encoder fields and leaves eval_idx at the last layer.
        decoder = self.model.decoder
        neck = self.model.encoder(self.model.backbone(x))
        layers = []
        for k in range(decoder.decoder.num_layers):
            decoder.decoder.eval_idx = k
            out = decoder(neck)
            logits = out['pred_logits'][0]
            layers.append(dict(score=torch.sigmoid(logits).cpu().numpy(), box=out['pred_boxes'][0].cpu().numpy(), raw=self.raw[k][0].cpu().numpy(), logit=logits.cpu().numpy(), ref=out['ref_points'][0].cpu().numpy()))
        return stages, enc, layers, out


def query_index_row(layers, box):
    final = layers[-1]
    iou = iou_matrix(xyxy(final['box']), box[None])[:, 0]
    good = iou >= .5
    tank = final['score'][:, TANK]
    return int(np.flatnonzero(good)[np.argmax(tank[good])]) if good.any() else int(iou.argmax()), iou


def native_delta(box, gt, scale):
    return [round(float(v), 2) for v in (((box[0] + box[2]) - (gt[0] + gt[2])) / 2 / scale, ((box[1] + box[3]) - (gt[1] + gt[3])) / 2 / scale, ((box[2] - box[0]) - (gt[2] - gt[0])) / scale, ((box[3] - box[1]) - (gt[3] - gt[1])) / scale)]


def follow(query, enc, layers, gt, scale, names):
    """One query from the encoder to the last decoder layer."""
    cell = int(enc['selected'][query])
    start = 0
    for n, stride in zip(enc['counts'], (8, 16, 32)):
        if cell < start + n:
            h, w = enc['shapes'][(8, 16, 32).index(stride)]
            i, j = np.unravel_index(cell - start, (h, w))
            level, anchor = stride, [float((j + .5) * stride), float((i + .5) * stride)]
            break
        start += n
    probability = enc['prob'][cell]
    trail = []
    boxes = [('encoder', xyxy(enc['boxes'][cell]), float(probability[TANK]), int(probability.argmax()), float(probability.max()), None, None),
             ('pre', xyxy(layers[0]['ref'][query]), None, None, None, None, None)]
    for k, layer in enumerate(layers):
        scores = layer['score'][query]
        boxes.append((f'L{k}', xyxy(layer['box'][query]), float(scores[TANK]), int(scores.argmax()), float(scores.max()), float(layer['raw'][query, TANK]), float(layer['logit'][query, TANK] - layer['raw'][query, TANK])))
    for name, box, tank, top, top_score, raw, lqe in boxes:
        trail.append(dict(step=name, box=[round(float(v), 1) for v in box], iou=round(float(iou_matrix(box[None], gt[None])[0, 0]), 3), tank=None if tank is None else round(tank, 4),
                          top_class=None if top is None else names[top], top_score=None if top_score is None else round(top_score, 4), raw_logit=None if raw is None else round(raw, 3), lqe_delta=None if lqe is None else round(lqe, 3), native_delta=native_delta(box, gt, scale)))
    return dict(query=query, encoder_rank=query, level=level, anchor_center=anchor, trail=trail)


def encoder_view(box, enc, names):
    rank, prob, boxes = enc['rank'], enc['prob'], xyxy(enc['boxes'])
    centers, levels = [], []
    for (h, w), stride in zip(enc['shapes'], (8, 16, 32)):
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        centers.append(np.stack([(xx.ravel() + .5) * stride, (yy.ravel() + .5) * stride], 1))
        levels.append(np.full(h * w, stride))
    centers, levels = np.concatenate(centers), np.concatenate(levels)
    inside = np.flatnonzero((centers[:, 0] >= box[0]) & (centers[:, 0] < box[2]) & (centers[:, 1] >= box[1]) & (centers[:, 1] < box[3]))
    iou = iou_matrix(boxes, box[None])[:, 0]
    rows = []
    for c in inside[np.argsort(rank[inside])][:10]:
        rows.append(dict(level=int(levels[c]), rank=int(rank[c]), selected=bool(rank[c] < TOP), tank=round(float(prob[c, TANK]), 4), top_class=names[int(prob[c].argmax())], top_score=round(float(prob[c].max()), 4), box_iou=round(float(iou[c]), 3)))
    best = int(iou.argmax())
    selected = enc['selected']
    near = np.abs(centers[selected] - (box[:2] + box[2:]) / 2).max(1)
    return dict(cells_in_gt=rows, cell_count_in_gt=len(inside), cutoff_top_class_prob=round(float(prob[selected[-1]].max()), 4),
                best_iou_cell=dict(iou=round(float(iou[best]), 3), rank=int(rank[best]), selected=bool(rank[best] < TOP), level=int(levels[best]), tank=round(float(prob[best, TANK]), 4), top_class=names[int(prob[best].argmax())], top_score=round(float(prob[best].max()), 4)),
                best_iou_selected=round(float(iou[selected].max()), 3), selected_within_16px=int((near <= 16).sum()), selected_within_32px=int((near <= 32).sum()),
                top300_by_level=dict(Counter(str(int(levels[c])) for c in selected)), top300_by_class=dict(Counter(names[int(prob[c].argmax())] for c in selected))), centers, levels


def tracked_queries(status, layers, box):
    final = layers[-1]
    iou = iou_matrix(xyxy(final['box']), box[None])[:, 0]
    tank = final['score'][:, TANK]
    picks = []
    good = iou >= .5
    if good.any():
        picks.append(('best-scoring box with IoU>=.5', int(np.flatnonzero(good)[np.argmax(tank[good])])))
    wrong = (iou >= .1) & (iou < .5)
    if wrong.any():
        picks.append(('best-scoring box with IoU .1-.5', int(np.flatnonzero(wrong)[np.argmax(tank[wrong])])))
    if not picks:
        centers = (xyxy(final['box'])[:, :2] + xyxy(final['box'])[:, 2:]) / 2
        distance = np.linalg.norm(centers - (box[:2] + box[2:]) / 2, axis=1)
        picks.append(('nearest final box', int(distance.argmin())))
    return picks[:2]


def neighbours(layers, box):
    final = layers[-1]
    boxes = xyxy(final['box'])
    iou = iou_matrix(boxes, box[None])[:, 0]
    tank = final['score'][:, TANK]
    keep = np.flatnonzero(iou >= .1)
    return [dict(query=int(q), iou=round(float(iou[q]), 3), tank=round(float(tank[q]), 4)) for q in keep[np.argsort(-tank[keep])][:6]]


def population(run, out, device, tracer, names):
    """Same decomposition for every GT, so a single sample can be placed against its group."""
    import torch
    annotations, _, _ = load_val(out)
    images = {i['id']: i for i in annotations['images']}
    rows = json.load(open(out / 'gt_rows.json'))
    by = defaultdict(list)
    for r in rows:
        by[r['image_id']].append(r)
    collected = defaultdict(list)
    with torch.no_grad():
        for count, (image_id, part) in enumerate(by.items()):
            x, (w, _) = to_input(ROOT / 'JPEGImages/val' / images[image_id]['file_name'])
            _, enc, layers, _ = tracer.run(x[None].to(device))
            s = INPUT / w
            for r in part:
                if r['native_gw'] <= 0 or r['native_gh'] <= 0:
                    continue
                box = np.array([r['gt_x1'], r['gt_y1'], r['gt_x2'], r['gt_y2']], np.float32) * s
                query, iou = query_index_row(layers, box)
                final = layers[-1]
                group = f"{r['source']}|{r['status']}|{strict_stage(r)}"
                row = dict(raw=float(final['raw'][query, TANK]), lqe=float(final['logit'][query, TANK] - final['raw'][query, TANK]), score=float(final['score'][query, TANK]), iou=float(iou[query]),
                           enc=float(enc['prob'][enc['selected'][query], TANK]), rank=int(query), l0=float(layers[0]['score'][query, TANK]), top_other=float(np.delete(final['score'][query], TANK).max()),
                           wrong_score=float(final['score'][(iou >= .1) & (iou < .5), TANK].max()) if ((iou >= .1) & (iou < .5)).any() else 0., count_good=int((iou >= .5).sum()))
                collected[group].append(row)
            print('population', count + 1, len(by), flush=True)
    return {group: dict(n=len(part), **{k: round(float(np.median([p[k] for p in part])), 3) for k in part[0]}) for group, part in collected.items()}


def draw_points(rgb, centers, levels, selected, x0, y0, span, display, highlight=()):
    image = Image.fromarray(rgb).convert('RGB')
    draw = ImageDraw.Draw(image)
    factor = display / span
    for c in selected:
        x, y = (centers[c][0] - x0) * factor, (centers[c][1] - y0) * factor
        if -4 <= x <= display + 4 and -4 <= y <= display + 4:
            radius = 3 if span < 300 else 1
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=LEVEL_COLOURS[int(levels[c])], width=1)
    return image


def boxed(image, boxes, x0, y0, factor, title):
    draw = ImageDraw.Draw(image)
    for box, color, width in boxes:
        draw.rectangle([(box[0] - x0) * factor, (box[1] - y0) * factor, (box[2] - x0) * factor, (box[3] - y0) * factor], outline=color, width=width)
    left, top, right, bottom = draw.textbbox((0, 0), title, font=SMALL)
    draw.rectangle([0, 0, right - left + 6, bottom - top + 5], fill=(0, 0, 0))
    draw.text((3, 2 - top), title, fill=(240, 240, 240), font=SMALL)
    return image


def figure(name, item, rgb, maps, bounds, enc, layers, centers, levels, traced, other_boxes, result, s, path):
    row = item['row']
    box = np.array([row['gt_x1'], row['gt_y1'], row['gt_x2'], row['gt_y2']], np.float32) * s
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    def window(span):
        return int(np.clip(cx - span / 2, 0, INPUT - span)), int(np.clip(cy - span / 2, 0, INPUT - span))

    gray = (rgb[..., 0] * 255).astype(np.uint8)

    def input_view(span):
        x0, y0 = window(span)
        v = (sampled(rgb[..., 0], 1, x0, y0, span, CELL) * 255).astype(np.uint8)
        return np.stack([v] * 3, -1), x0, y0, CELL / span

    x0, y0 = window(BIG_SPAN)
    factor = CELL / BIG_SPAN
    marks = [(box, GT_C, 1)] + [(b, (0, 130, 60), 1) for b in other_boxes]

    def stage_panel(stage):
        low, high = bounds[stage]
        values = sampled(maps[stage], STRIDES[stage], x0, y0, BIG_SPAN, CELL)
        return panel(colour(values, low, high), f'{stage} L2 [{low:.1f},{high:.1f}]', marks, STRIDES[stage], x0, y0, factor)

    whole = np.asarray(Image.fromarray(np.stack([gray] * 3, -1)).resize((CELL, CELL), Image.BILINEAR))
    big_rgb, _, _, _ = input_view(BIG_SPAN)
    selected = enc['selected']
    topk_whole = draw_points(whole, centers, levels, selected, 0, 0, INPUT, CELL)
    boxed(topk_whole, [(box, GT_C, 1)], 0, 0, CELL / INPUT, 'Top-300 cells, whole image')
    topk_zoom = draw_points(big_rgb, centers, levels, selected, x0, y0, BIG_SPAN, CELL)
    boxed(topk_zoom, marks, x0, y0, factor, 'Top-300 (blue L8, yellow L16, pink L32)')
    best_level = 8 if traced[0]['level'] == 8 else traced[0]['level']
    prob_name = {8: 'T3', 16: 'T4', 32: 'T5'}[best_level]
    prob_panel = panel(colour(sampled(maps[prob_name], STRIDES[prob_name], x0, y0, BIG_SPAN, CELL), 0, 1), f'{prob_name} P(tank) [0,1]', marks, STRIDES[prob_name], x0, y0, factor)

    rows = [[panel(whole, 'input whole', [(box, GT_C, 1)], None, 0, 0, CELL / INPUT), panel(big_rgb, f'input x{factor:.1f}', marks, None, x0, y0, factor), stage_panel('P3'), stage_panel('P4'), stage_panel('P5')],
            [stage_panel('F3'), stage_panel('F4'), topk_whole, topk_zoom, prob_panel]]
    zoom_rgb, zx, zy, zf = input_view(BOX_SPAN)
    steps = ['encoder', 'pre', 'L0', 'L1', 'L2', 'L3']
    decoded = []
    for step in steps:
        image = Image.fromarray(zoom_rgb).convert('RGB')
        marks_step = [(box, GT_C, 1)]
        title = [step]
        for t, track in enumerate(traced):
            line = next(x for x in track['trail'] if x['step'] == step)
            marks_step.append((line['box'], TRACK_COLOURS[t], 1))
            title.append(f"{'pink' if t == 0 else 'cyan'} {line['iou']:.2f}/{'-' if line['tank'] is None else format(line['tank'], '.2f')}")
        decoded.append(boxed(image, marks_step, zx, zy, zf, ' '.join(title)))
    final_image = Image.fromarray(zoom_rgb).convert('RGB')
    labels, scores, boxes_ = result
    keep = (labels == TANK) & (scores >= .05)
    final_marks = [(box, GT_C, 1)] + [(b * s, (70, 190, 255) if v >= SCORE else (150, 110, 255), 1) for b, v in zip(boxes_[keep], scores[keep])]
    boxed(final_image, final_marks, zx, zy, zf, 'final tank >=.05 (blue >=.3)')
    rows.append(decoded[:5])
    rows.append([decoded[5], final_image])
    text = Image.new('RGB', (CELL * 3 + 10, CELL), (16, 16, 16))
    draw = ImageDraw.Draw(text)
    lines = [f"{name}: {item['status']} ({item['stage']})", f"GT native {row['native_gw']:.0f}x{row['native_gh']:.0f}, input {row['input_gw']:.1f}x{row['input_gh']:.1f}, nearest GT {row['nn_input']:.0f}px"]
    for t, track in enumerate(traced):
        last, first = track['trail'][-1], track['trail'][0]
        lines.append(f"{'pink' if t == 0 else 'cyan'} query {track['query']} (rank {track['encoder_rank']}), L{track['level']}")
        lines.append(f"  IoU {first['iou']:.3f} -> {last['iou']:.3f}, tank {first['tank']:.3f} -> {last['tank']:.3f}")
        lines.append(f"  top class {last['top_class']} {last['top_score']:.3f}; raw logit {last['raw_logit']:.2f}, LQE {last['lqe_delta']:+.2f}")
        lines.append(f"  residual native dx dy dw dh {last['native_delta']}")
    for i, line in enumerate(lines):
        draw.text((6, 6 + i * 17), line, fill=(235, 235, 235), font=FONT)
    rows[-1].append(text)

    canvas = Image.new('RGB', (CELL * 5 + 20, 24 + (CELL + 5) * len(rows)), (24, 24, 24))
    ImageDraw.Draw(canvas).text((5, 4), f"{name}  {row['file_name']}  gt {row['gt_id']}  box {box.round(1).tolist()} (input px). Boxes: green GT; title IoU/tank.", fill=(240, 240, 240), font=FONT)
    for r, cells in enumerate(rows):
        px = 0
        for c in cells:
            canvas.paste(c, (px, 24 + r * (CELL + 5)))
            px += c.width + 5
    canvas.save(path)
    small = {stage: stage_panel(stage).resize((150, 150), Image.NEAREST) for stage in STAGES}
    small['input'] = panel(big_rgb, name, marks, None, x0, y0, factor).resize((150, 150), Image.BILINEAR)
    small['topk'] = topk_zoom.resize((150, 150), Image.BILINEAR)
    small['encoder'] = decoded[0].resize((150, 150), Image.BILINEAR)
    small['L3'] = decoded[5].resize((150, 150), Image.BILINEAR)
    return small


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--image', default='0112914.png')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    folder = args.out / 'trace'
    folder.mkdir(exist_ok=True)
    rows = json.load(open(args.out / 'gt_rows.json'))
    chosen = choose(rows, args.image)
    annotations, _, grouped = load_val(args.out)
    names = {c['id']: c['name'] for c in annotations['categories']}
    bounds = json.load(open(args.out / 'display_scales.json'))
    image_info = next(i for i in annotations['images'] if i['file_name'] == args.image)
    model, post = load_model(args.run, args.device)
    tracer = Tracer(model)
    x, (w, h) = to_input(ROOT / 'JPEGImages/val' / args.image)
    s = INPUT / w
    with torch.no_grad():
        stages, enc, layers, last = tracer.run(x[None].to(args.device))
        result = post(last, torch.tensor([[w, h]], device=args.device, dtype=torch.float32))[0]
    labels, scores, boxes = result['labels'].cpu().numpy().astype(int), result['scores'].cpu().numpy(), result['boxes'].cpu().numpy()
    targets = grouped[image_info['id']]
    g_cls = np.array([a['category_id'] for a in targets], int)
    g_box = np.array([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in targets], np.float32)
    keep, kinds, _, fn_kind, _, taken = match_image(labels, scores, boxes, g_cls, g_box, SCORE, .5)
    stages['input'] = x[:1].numpy()
    maps = {k: l2_map(v) for k, v in stages.items()}
    rgb = x.numpy().transpose(1, 2, 0)
    report = dict(image=args.image, input=INPUT, image_kinds=dict(Counter(k for k in fn_kind if k)), image_tp=int(sum(k is None for k in fn_kind)), samples={})
    sheet = {}
    for name, item in chosen.items():
        row = item['row']
        box = np.array([row['gt_x1'], row['gt_y1'], row['gt_x2'], row['gt_y2']], np.float32) * s
        others = [np.array([a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]], np.float32) * s for a in targets if a['id'] != row['gt_id']]
        region = {stage: {k: round(v, 3) for k, v in region_stats(stages[stage], STRIDES[stage], box, np.array(others).reshape(-1, 4)).items()} for stage in ('input', *STAGES)}
        view, centers, levels = encoder_view(box, enc, names)
        traced = [dict(follow(q, enc, layers, box, s, names), role=role) for role, q in tracked_queries(item['status'], layers, box)]
        index = next(i for i, a in enumerate(targets) if a['id'] == row['gt_id'])
        final = dict(kind=fn_kind[index] or 'TP')
        if taken[index] >= 0:
            d = keep[taken[index]]
            final.update(score=round(float(scores[d]), 4), iou=round(float(iou_matrix(boxes[d][None], g_box[index][None])[0, 0]), 3))
        report['samples'][name] = dict(status=item['status'], stage=item['stage'], group_size=item['group_size'], group_median=item['group_median'], candidates_in_image=item['candidates'], gt_id=row['gt_id'], native=[row['native_gw'], row['native_gh']], input=[row['input_gw'], row['input_gh']], nn_input=row['nn_input'],
                                       box_input=[round(float(v), 1) for v in box], fields={f: round(row[f], 3) for f in FIELDS}, region=region, encoder=view, queries=traced, final=final, neighbours=neighbours(layers, box))
        sheet[name] = figure(name, item, rgb, maps, bounds, enc, layers, centers, levels, traced, others, (labels, scores, boxes), s, folder / f'{name}_{item["status"]}.png')
    columns = ('input', 'P3', 'P4', 'P5', 'F3', 'F4', 'topk', 'encoder', 'L3')
    canvas = Image.new('RGB', (153 * len(columns) + 8, 20 + 153 * len(sheet)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for c, label in enumerate(columns):
        draw.text((8 + c * 153, 3), label, fill=(230, 230, 230), font=FONT)
    for r, name in enumerate(sheet):
        for c, label in enumerate(columns):
            canvas.paste(sheet[name][label], (8 + c * 153, 20 + r * 153))
    canvas.save(folder / 'compare_four.png')
    report['population'] = population(args.run, args.out, args.device, tracer, names)
    (folder / 'trace.json').write_text(json.dumps(report, indent=1, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))
    print('done', flush=True)


if __name__ == '__main__':
    main()
