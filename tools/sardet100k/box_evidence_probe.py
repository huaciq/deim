"""D8: image and multi-scale evidence of localisation quality, measured on two candidates of the same GT. No training of
the detector, no change to it.

D8.1  For one GT, compare its better-localised candidate with a worse one. Both come from the same image, the same
      target and the same class, so scene and class differences cannot explain a difference between them.
D8.2  Four independent families of box-relative features, each probed alone:
        P1  interior scattering structure   7x7 bilinear samples of the original image inside the box, their mean,
                                            std and max, and the offset of the bright-excess centroid from the box
                                            centre (in box units)
        P2  box boundary                    per edge (L, R, T, B) a profile across the edge at -.25, -.125, 0, +.125,
                                            +.25 box units (positive = inside), averaged along the edge; per-edge
                                            inside-minus-outside contrast; share of the bright excess of the 2x box
                                            region that falls inside the box
        P3  surrounding background          ring between 1.5x and 3x the box: mean, std, median, q90, max; mean of each
                                            side; ring max - interior max; ring mean - interior mean; share of ring
                                            samples brighter than the interior mean
        P4  multi-scale network features    for backbone stage 1 (stride 4, 96 ch) and the three encoder outputs
                                            (stride 8/16/32, 256 ch): box mean of 3x3 bilinear samples, and a 5x5 grid
                                            of the feature L2 norm over the 2x box region, divided by its mean.
                                            P4 = all four scales concatenated; P4_s4, P4_p3, P4_p4, P4_p5 = one scale
      Image features use the original image (8-bit grey, original resolution, values / 255); network features the
      640 input of the same forward pass. No feature uses the GT.

Candidates and pairs are those of D7 (quality_signal_probe.py): queries whose highest-IoU same-class GT is this GT,
IoU >= .5, at least two per GT. The records also hold S0 (box geometry), S1_final (the logit the postprocessor ranks)
and S4_h1 (box head first hidden layer), so the probes, the conditional gains and the swap oracle run unchanged:
  quality_signal_probe.py joint FEATURES --signals P1 S0+S1+P1 ...
  quality_swap_oracle.py FEATURES OUT_DIR --signals S0+S1+P1 ...

Commands:
  extract RUN OUT --checkpoint PATH [--threads 16]       (server, CPU) OUT/probe_features.pt, OUT/queries.npz
  contrast FEATURES --out DIR                            D8.1 table: better minus worse candidate of the same GT
  gallery RUN FEATURES --out DIR --checkpoint PATH       D8.1 figure (server: needs images and weights)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import GROUP_ORDER, group_of, iou_xyxy, r3, sigmoid  # noqa: E402

SCALES = ['s4', 'p3', 'p4', 'p5']
GRID_IN = 7
EDGE_T = np.array([-.25, -.125, 0., .125, .25])
EDGE_S = np.array([.1, .3, .5, .7, .9])
# Interpretable descriptors for D8.1 (one scalar each, higher/lower is readable).
DESCRIPTORS = ['interior_mean', 'interior_max', 'centroid_offset', 'edge_contrast', 'energy_capture',
               'ring_mean', 'ring_std', 'ring_max_minus_interior_max', 'ring_bright_share',
               'net_s4_inside_share', 'net_p3_inside_share', 'net_p4_inside_share', 'net_p5_inside_share']


# ---------------------------------------------------------------- box-relative sampling
def sample(fmap, pts):
    """fmap: tensor (C, H, W); pts: (N, P, 2) normalised (x, y) in [0, 1] of the image. Returns (N, P, C) numpy.
    align_corners=False: 0 and 1 are the outer edges of the first and last pixel, as are box coordinates."""
    import torch
    import torch.nn.functional as F
    grid = torch.as_tensor(pts * 2 - 1, dtype=fmap.dtype)[None]
    out = F.grid_sample(fmap[None], grid, mode='bilinear', padding_mode='border', align_corners=False)
    return out[0].permute(1, 2, 0).numpy()


def box_points(box, u, v):
    """box (N, 4) normalised xyxy; u, v (P,) box coordinates (0 = left/top edge, 1 = right/bottom edge) -> (N, P, 2)."""
    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    return np.stack([box[:, None, 0] + u[None] * w[:, None], box[:, None, 1] + v[None] * h[:, None]], -1)


def mesh(values):
    u, v = np.meshgrid(values, values)
    return u.ravel(), v.ravel()


def image_features(gray, box):
    """gray: tensor (1, H, W) in [0, 1]; box: (N, 4). Returns P1 (N, 54), P2 (N, 25), P3 (N, 12), descriptor dict."""
    n = len(box)
    # P1 interior
    u, v = mesh((np.arange(GRID_IN) + .5) / GRID_IN)
    inner = sample(gray, box_points(box, u, v))[..., 0]                                          # (N, 49)
    # P3 ring between 1.5x and 3x the box; also gives the background level for the excess
    ru, rv = mesh((np.arange(12) + .5) / 12 * 3 - 1)                                             # box coords -1 .. 2
    ring_mask = (np.abs(ru - .5) > .75) | (np.abs(rv - .5) > .75)
    ring = sample(gray, box_points(box, ru[ring_mask], rv[ring_mask]))[..., 0]
    ring_median = np.median(ring, 1)
    excess_in = np.maximum(inner - ring_median[:, None], 0)
    wsum = excess_in.sum(1) + 1e-6
    cx = ((excess_in * (u - .5)).sum(1) / wsum)
    cy = ((excess_in * (v - .5)).sum(1) / wsum)
    p1 = np.concatenate([inner, inner.mean(1, keepdims=True), inner.std(1, keepdims=True), inner.max(1, keepdims=True),
                         cx[:, None], cy[:, None]], 1)
    # P2 boundary profiles, inward positive
    s_rep, t_rep = np.repeat(EDGE_S, len(EDGE_T)), np.tile(EDGE_T, len(EDGE_S))
    edges = {'L': (t_rep, s_rep), 'R': (1 - t_rep, s_rep), 'T': (s_rep, t_rep), 'B': (s_rep, 1 - t_rep)}
    profiles, contrasts = [], []
    for eu, ev in edges.values():
        prof = sample(gray, box_points(box, eu, ev))[..., 0].reshape(n, len(EDGE_S), len(EDGE_T)).mean(1)
        profiles.append(prof)
        contrasts.append(prof[:, EDGE_T > 0].mean(1) - prof[:, EDGE_T < 0].mean(1))
    eu2, ev2 = mesh((np.arange(12) + .5) / 12 * 2 - .5)                                          # box coords -.5 .. 1.5
    region = np.maximum(sample(gray, box_points(box, eu2, ev2))[..., 0] - ring_median[:, None], 0)
    inside = (eu2 >= 0) & (eu2 <= 1) & (ev2 >= 0) & (ev2 <= 1)
    capture = region[:, inside].sum(1) / (region.sum(1) + 1e-6)
    contrasts = np.stack(contrasts, 1)
    p2 = np.concatenate(profiles + [contrasts, capture[:, None]], 1)
    # P3 statistics
    sides = [ru[ring_mask] < -.25, ru[ring_mask] > 1.25, rv[ring_mask] < -.25, rv[ring_mask] > 1.25]
    side_means = np.stack([ring[:, s].mean(1) for s in sides], 1)
    imean, imax = inner.mean(1), inner.max(1)
    p3 = np.concatenate([np.stack([ring.mean(1), ring.std(1), ring_median, np.quantile(ring, .9, axis=1), ring.max(1)], 1),
                         side_means, np.stack([ring.max(1) - imax, ring.mean(1) - imean, (ring > imean[:, None]).mean(1)], 1)], 1)
    desc = dict(interior_mean=imean, interior_max=imax, centroid_offset=np.hypot(cx, cy), edge_contrast=contrasts.mean(1),
                energy_capture=capture, ring_mean=ring.mean(1), ring_std=ring.std(1),
                ring_max_minus_interior_max=ring.max(1) - imax, ring_bright_share=(ring > imean[:, None]).mean(1))
    return p1, p2, p3, desc


def network_features(fmap, box):
    """fmap: tensor (C, h, w) at the 640 input. Returns (N, C + 25) and the inside share of the feature norm."""
    import torch
    u, v = mesh((np.arange(3) + .5) / 3)
    roi = sample(fmap, box_points(box, u, v)).mean(1)                                             # (N, C)
    norm = torch.linalg.vector_norm(fmap, dim=0, keepdim=True)
    gu, gv = mesh((np.arange(5) + .5) / 5 * 2 - .5)                                              # box coords -.5 .. 1.5
    energy = sample(norm, box_points(box, gu, gv))[..., 0]
    energy = energy / (energy.mean(1, keepdims=True) + 1e-6)
    inside = (gu >= 0) & (gu <= 1) & (gv >= 0) & (gv <= 1)
    return np.concatenate([roi, energy], 1), energy[:, inside].mean(1) / (energy[:, ~inside].mean(1) + 1e-6)


# ---------------------------------------------------------------- extract (server, CPU)
def extract(args):
    import torch
    from PIL import Image
    from error_gallery import ROOT
    from tank_feature_diagnosis import load_val, to_input, xyxy
    from tank_matching_counterfactual import Runner
    torch.set_num_threads(args.threads)
    annotations, source, grouped = load_val(args.out)
    name_of = {c['id']: c['name'] for c in annotations['categories']}
    runner = Runner(args.run, 'cpu', args.checkpoint)
    model, inner = runner.model, runner.inner
    last = runner.default_idx
    grab = {}
    inner.lqe_layers[last].register_forward_hook(lambda m, inp, out: grab.update(final=out.detach()))
    bbox_head = model.decoder.dec_bbox_head[last]
    bbox_head.layers[0].register_forward_hook(lambda m, inp, out: grab.update(h1=bbox_head.act(out).detach()))
    model.backbone.stages[0].register_forward_hook(lambda m, inp, out: grab.update(s4=out.detach()))
    model.encoder.register_forward_hook(lambda m, inp, out: grab.update(enc=[t.detach() for t in (out[0] if isinstance(out, tuple) else out)]))
    records, ids, logits_all, boxes_all = [], [], [], []
    images = annotations['images']
    with torch.no_grad():
        for n, image in enumerate(images):
            g = group_of(source[image['id']], image['width'], image['height'])
            anns = grouped[image['id']]
            path = ROOT / 'JPEGImages/val' / image['file_name']
            x, _ = to_input(path)
            inner.eval_idx = last
            result = model(x[None])
            boxes_c = result['pred_boxes'][0].float().numpy()
            boxes = xyxy(boxes_c, 1)
            final = grab['final'][0].float().numpy()
            assert np.allclose(final, result['pred_logits'][0].float().numpy(), atol=1e-4)
            ids.append(image['id'])
            logits_all.append(final)
            boxes_all.append(boxes)
            if not anns:
                continue
            _, xy = runner.targets(image, anns)
            labels = np.array([a['category_id'] for a in anns])
            gray = torch.from_numpy(np.asarray(Image.open(path).convert('L'), np.float32) / 255.)[None]
            maps = dict(s4=grab['s4'][0], p3=grab['enc'][0][0], p4=grab['enc'][1][0], p5=grab['enc'][2][0])
            for c in np.unique(labels):
                idx = np.flatnonzero(labels == c)
                iou = iou_xyxy(boxes, xy[idx])
                owner, best = iou.argmax(1), iou.max(1)
                for j, t in enumerate(idx):
                    cand = np.flatnonzero((owner == j) & (best >= .5))
                    if len(cand) < 2:
                        continue
                    b = boxes[cand].astype(np.float64)
                    p1, p2, p3, desc = image_features(gray, b)
                    rec = dict(image=image['id'], cell=f'{g}/{name_of[int(c)]}', gt=int(t), gt_box=xy[t].astype(np.float32), query=cand,
                               iou=iou[cand, j].astype(np.float32), score=sigmoid(final[cand, c]).astype(np.float32),
                               S0=np.concatenate([boxes_c[cand], np.log(np.maximum(boxes_c[cand, 2:], 1e-6))], 1).astype(np.float32),
                               S1_final=final[cand, c].astype(np.float32), S4_h1=grab['h1'][0, cand].float().numpy(),
                               P1=p1.astype(np.float32), P2=p2.astype(np.float32), P3=p3.astype(np.float32))
                    for s in SCALES:
                        feat, share = network_features(maps[s].float(), b)
                        rec[f'P4_{s}'] = feat.astype(np.float32)
                        desc[f'net_{s}_inside_share'] = share
                    rec['P4'] = np.concatenate([rec[f'P4_{s}'] for s in SCALES], 1)
                    rec['D8'] = np.stack([desc[k] for k in DESCRIPTORS], 1).astype(np.float32)
                    records.append(rec)
            if (n + 1) % 500 == 0:
                print('images', n + 1, len(images), 'records', len(records), flush=True)
    np.savez_compressed(args.out / 'queries.npz', image_id=np.array(ids), logits=np.stack(logits_all).astype(np.float32),
                        boxes=np.stack(boxes_all).astype(np.float32))
    meta = dict(run=str(args.run), checkpoint=str(args.checkpoint), epoch=runner.epoch, layer=last, images=len(ids),
                descriptors=DESCRIPTORS, dims={k: int(records[0][k].shape[1]) for k in ['P1', 'P2', 'P3', 'P4'] + [f'P4_{s}' for s in SCALES]})
    torch.save(dict(meta=meta, records=records), args.out / 'probe_features.pt')
    print('written', args.out / 'probe_features.pt', len(records), meta['dims'], flush=True)


# ---------------------------------------------------------------- D8.1 contrast (local)
def score_wrong_pair(r):
    """(better, worse): the best-IoU candidate and the top-score candidate when they differ by IoU >= .1; else None."""
    best, top = int(np.argmax(r['iou'])), int(np.argmax(r['score']))
    return (best, top) if best != top and r['iou'][best] - r['iou'][top] >= .1 else None


def contrast(args):
    import torch
    data = torch.load(args.features, weights_only=False)
    names = data['meta']['descriptors']
    rng = np.random.default_rng(0)
    report = {}
    for cell in sorted({r['cell'] for r in data['records']}):
        recs = [r for r in data['records'] if r['cell'] == cell]
        sets = {'score_wrong': [], 'all_pairs': []}
        for r in recs:
            p = score_wrong_pair(r)
            if p:
                sets['score_wrong'].append((r['image'], r['D8'][p[0]] - r['D8'][p[1]], 1.))
            n = len(r['iou'])
            pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if max(r['score'][a], r['score'][b]) >= .1 and r['iou'][a] != r['iou'][b]]
            for a, b in pairs:
                hi, lo = (a, b) if r['iou'][a] > r['iou'][b] else (b, a)
                sets['all_pairs'].append((r['image'], r['D8'][hi] - r['D8'][lo], 1. / len(pairs)))
        out = dict(gts=len(recs))
        for key, rows in sets.items():
            if len(rows) < 20:
                out[key] = dict(n=len(rows))
                continue
            img = np.array([x[0] for x in rows])
            diff = np.stack([x[1] for x in rows])
            w = np.array([x[2] for x in rows])
            pos = (diff > 0).astype(float)
            images = np.unique(img)
            by = {i: np.flatnonzero(img == i) for i in images}
            boots = []
            for _ in range(1000):
                idx = np.concatenate([by[i] for i in rng.choice(images, len(images))])
                boots.append((w[idx, None] * pos[idx]).sum(0) / w[idx].sum())
            boots = np.array(boots)
            share = (w[:, None] * pos).sum(0) / w.sum()
            out[key] = dict(n=len(rows), images=len(images), descriptors={
                k: dict(median_diff=r3(np.median(diff[:, i])), share_better_higher=r3(share[i]),
                        ci95=[r3(np.percentile(boots[:, i], 2.5)), r3(np.percentile(boots[:, i], 97.5))]) for i, k in enumerate(names)})
        report[cell] = out
        print(cell, {k: v.get('n') for k, v in out.items() if isinstance(v, dict)}, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'contrast.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('written', args.out / 'contrast.json')


# ---------------------------------------------------------------- D8.1 gallery (server)
def gallery(args):
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from error_gallery import ROOT
    from tank_feature_diagnosis import load_val, to_input, xyxy
    from tank_matching_counterfactual import Runner
    torch.set_num_threads(args.threads)
    data = torch.load(args.features, weights_only=False)
    annotations, _, _ = load_val(args.out)
    info = {i['id']: i for i in annotations['images']}
    name_id = {c['name']: c['id'] for c in annotations['categories']}
    runner = Runner(args.run, 'cpu', args.checkpoint)
    model = runner.model
    grab = {}
    model.backbone.stages[0].register_forward_hook(lambda m, inp, out: grab.update(s4=out.detach()))
    model.encoder.register_forward_hook(lambda m, inp, out: grab.update(enc=[t.detach() for t in (out[0] if isinstance(out, tuple) else out)]))
    font = ImageFont.load_default(size=13)
    panel, cols = 180, 6
    rng = np.random.default_rng(0)
    rows = []
    for cell, k in [c.split('=') for c in args.cells]:
        cand = [r for r in data['records'] if r['cell'] == cell and score_wrong_pair(r)]
        cand.sort(key=lambda r: -float(r['iou'].max() - r['iou'][int(np.argmax(r['score']))]))
        pool = cand[:4 * int(k)]
        pick = [pool[i] for i in sorted(rng.choice(len(pool), min(int(k), len(pool)), replace=False))]
        rows += [(cell, r) for r in pick]
    sheet = Image.new('RGB', (cols * panel, len(rows) * (panel + 18)), 'white')
    draw_sheet = ImageDraw.Draw(sheet)
    for row, (cell, r) in enumerate(rows):
        image = info[r['image']]
        path = ROOT / 'JPEGImages/val' / image['file_name']
        x, _ = to_input(path)
        with torch.no_grad():
            result = model(x[None])
        all_boxes = xyxy(result['pred_boxes'][0].float().numpy(), 1)
        score = torch.sigmoid(result['pred_logits'][0, :, name_id[cell.split('/')[1]]]).numpy()
        gt = r['gt_box']
        iou_all = iou_xyxy(all_boxes, gt[None])[:, 0]
        better, worse = score_wrong_pair(r)
        # locate the two candidates by their box (query indices of a second CPU pass can differ at the top-300 margin)
        geo = r['S0'][:, :4].astype(np.float64)
        rec_boxes = np.concatenate([geo[:, :2] - geo[:, 2:] / 2, geo[:, :2] + geo[:, 2:] / 2], 1)
        qb, qw = (int(np.abs(all_boxes - rec_boxes[i]).max(1).argmin()) for i in (better, worse))
        low = np.flatnonzero((iou_all >= .1) & (iou_all < .5))
        ql = int(low[np.argmax(score[low])]) if len(low) else None
        # crop: 3x the GT extent, at least 24 original pixels
        W, H = image['width'], image['height']
        gw, gh = (gt[2] - gt[0]) * W, (gt[3] - gt[1]) * H
        side = max(3 * max(gw, gh), 24)
        cxp, cyp = (gt[0] + gt[2]) / 2 * W, (gt[1] + gt[3]) / 2 * H
        crop = np.array([cxp - side / 2, cyp - side / 2, cxp + side / 2, cyp + side / 2])

        def to_panel(b):
            return [(b[0] * W - crop[0]) / side * panel, (b[1] * H - crop[1]) / side * panel,
                    (b[2] * W - crop[0]) / side * panel, (b[3] * H - crop[1]) / side * panel]

        gray = Image.open(path).convert('L')
        base = gray.crop(tuple(crop.round().astype(int))).resize((panel, panel), Image.NEAREST).convert('RGB')
        tiles = [base.copy(), base.copy()]
        d0 = ImageDraw.Draw(tiles[0])
        d0.rectangle(to_panel(gt), outline=(255, 220, 0))
        d0.rectangle(to_panel(all_boxes[qb]), outline=(40, 220, 40))
        d0.rectangle(to_panel(all_boxes[qw]), outline=(255, 50, 50))
        if ql is not None:
            d0.rectangle(to_panel(all_boxes[ql]), outline=(255, 150, 0))
        # contrast-stretched crop without boxes, to see the scattering structure
        a = np.asarray(tiles[1], np.float32)
        a = (a - np.percentile(a, 2)) / (np.percentile(a, 99.5) - np.percentile(a, 2) + 1e-6)
        tiles[1] = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
        d1 = ImageDraw.Draw(tiles[1])
        d1.rectangle(to_panel(all_boxes[qb]), outline=(40, 220, 40))
        d1.rectangle(to_panel(all_boxes[qw]), outline=(255, 50, 50))
        maps = [grab['s4'][0]] + [t[0] for t in grab['enc']]
        for fm in maps:
            e = torch.linalg.vector_norm(fm.float(), dim=0).numpy()
            h, w = e.shape
            c = crop / np.array([W, H, W, H]) * np.array([w, h, w, h])
            img = Image.fromarray(((e - e.min()) / (e.max() - e.min() + 1e-6) * 255).astype(np.uint8))
            t = img.crop(tuple(c)).resize((panel, panel), Image.BILINEAR).convert('RGB')
            dt = ImageDraw.Draw(t)
            dt.rectangle(to_panel(all_boxes[qb]), outline=(40, 220, 40))
            dt.rectangle(to_panel(all_boxes[qw]), outline=(255, 50, 50))
            tiles.append(t)
        y = row * (panel + 18)
        for col, t in enumerate(tiles):
            sheet.paste(t, (col * panel, y + 18))
        label = (f'{cell} img {r["image"]}  better IoU {iou_all[qb]:.2f} s {score[qb]:.2f} | top-score IoU {iou_all[qw]:.2f} s {score[qw]:.2f}'
                 + (f' | IoU<.5 IoU {iou_all[ql]:.2f} s {score[ql]:.2f}' if ql is not None else ''))
        draw_sheet.text((4, y + 2), label, fill=(0, 0, 0), font=font)
    args.out.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out / 'd81_gallery.png')
    print('written', args.out / 'd81_gallery.png', len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract')
    e.add_argument('run', type=Path)
    e.add_argument('out', type=Path)
    e.add_argument('--checkpoint', type=Path, required=True)
    e.add_argument('--threads', type=int, default=16)
    c = sub.add_parser('contrast')
    c.add_argument('features')
    c.add_argument('--out', type=Path, required=True)
    g = sub.add_parser('gallery')
    g.add_argument('run', type=Path)
    g.add_argument('features')
    g.add_argument('--out', type=Path, required=True)
    g.add_argument('--checkpoint', type=Path, required=True)
    g.add_argument('--threads', type=int, default=16)
    g.add_argument('--cells', nargs='*', default=['ogsod/tank=8', 'ogsod/bridge=6', 'other_512/aircraft=6', 'other_256/ship=4'])
    args = parser.parse_args()
    if args.command == 'extract':
        args.out.mkdir(parents=True, exist_ok=True)
    {'extract': extract, 'contrast': contrast, 'gallery': gallery}[args.command](args)


if __name__ == '__main__':
    main()
