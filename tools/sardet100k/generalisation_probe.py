"""D12.2: does the generalisation gap arise in the features or in the detection heads? Frozen-feature probes.
No training of the detector, no change to it; train and val only.

For one checkpoint, features are pooled at every GT box (positives) and at one random box of the same size per GT that
overlaps no GT of the image (negatives), on the 12,000 train images of D12 and on all val images:
  IMG    the 24x24 grey crop of the 640 input, zero mean and unit norm (floor: what pixels alone give)
  BB     backbone outputs at stride 8 / 16 / 32, mean of 3x3 bilinear samples in the box, concatenated
  ENC    encoder outputs at stride 8 / 16 / 32 (256 channels each), same pooling, concatenated
  ENC_SCORE  the detector's own encoder classification logit: maximum over classes and over the encoder tokens whose
         centre lies in the box (nearest stride-8 token when none does). Not a probe: the trained readout itself.
A class-agnostic linear probe (logistic regression, target versus background) is fitted per feature on train split A
(two thirds of the 12,000 train images) and evaluated on three populations that differ in who has seen them:
  train B   the other third of the train images: seen by the DETECTOR in training, never by the probe
  val       seen by neither
Each population is split by `sim` (D12): similarity of the target to a GT of another train image.

Reading fixed before the run. For targets without a near-duplicate (sim < .9), compare the probe's recall on train B
with its recall on val, at the threshold that gives 1 % false positives on val negatives:
  * train B clearly above val  -> the features of images the detector trained on are target-like in a way that does
    not transfer: the gap is already in the representation (backbone / encoder);
  * train B about equal to val -> the features do not carry the memorisation; the detector's extra gap between
    train and val (D12: recall 92-97 % against 71-92 %) arises after the encoder, in the decoder and heads.
The seen/novel difference inside val alone is not evidence either way: novel targets may simply be less typical.

Usage (server; OUT directly under E000's independent_verify_20261106/, D12 its d12_generalisation directory):
  extract RUN OUT --d12 D12 [--device cuda:0]        OUT/features_{train,val}.npz
  analyse OUT --d12 D12 [--device cuda:0]            OUT/generalisation_probe.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import iou_xyxy, r3  # noqa: E402

FEATURES = ['IMG', 'BB', 'ENC']
SIM_BINS = [('sim>=.9', .9, 2.), ('sim .7-.9', .7, .9), ('sim <.7', -2., .7)]
CROP = 24


def roi_mean(fmap, boxes):
    """fmap (C, h, w) tensor; boxes (N, 4) normalised xyxy tensor on the same device -> (N, C)."""
    import torch
    import torch.nn.functional as F
    u = (torch.arange(3, device=fmap.device, dtype=torch.float32) + .5) / 3
    gx = boxes[:, None, 0] + u[None] * (boxes[:, 2] - boxes[:, 0])[:, None]      # N, 3
    gy = boxes[:, None, 1] + u[None] * (boxes[:, 3] - boxes[:, 1])[:, None]
    grid = torch.stack([gx[:, None, :].expand(-1, 3, -1), gy[:, :, None].expand(-1, -1, 3)], -1).reshape(1, len(boxes), 9, 2) * 2 - 1
    out = F.grid_sample(fmap[None].float(), grid, mode='bilinear', padding_mode='border', align_corners=False)   # 1, C, N, 9
    return out[0].mean(-1).T


def crops(image, boxes):
    """image (3, H, W) tensor in [0, 1]; boxes (N, 4) normalised xyxy -> (N, CROP*CROP), zero mean and unit norm."""
    import torch
    import torch.nn.functional as F
    u = (torch.arange(CROP, device=image.device, dtype=torch.float32) + .5) / CROP
    gx = boxes[:, None, 0] + u[None] * (boxes[:, 2] - boxes[:, 0])[:, None]
    gy = boxes[:, None, 1] + u[None] * (boxes[:, 3] - boxes[:, 1])[:, None]
    grid = torch.stack([gx[:, None, :].expand(-1, CROP, -1), gy[:, :, None].expand(-1, -1, CROP)], -1) * 2 - 1    # N, CROP, CROP, 2
    grey = image.mean(0, keepdim=True)[None].expand(len(boxes), -1, -1, -1)
    out = F.grid_sample(grey, grid, mode='bilinear', padding_mode='border', align_corners=False).reshape(len(boxes), -1)
    out = out - out.mean(1, keepdim=True)
    return out / out.norm(dim=1, keepdim=True).clamp(min=1e-6)


def negative_boxes(gt, rng, tries=20):
    """One box of each GT's size at a random place with zero overlap with every GT; rows of failures are dropped."""
    out = []
    for b in gt:
        w, h = b[2] - b[0], b[3] - b[1]
        for _ in range(tries):
            x, y = rng.uniform(0, max(1 - w, 1e-6)), rng.uniform(0, max(1 - h, 1e-6))
            cand = np.array([x, y, x + w, y + h])
            if iou_xyxy(cand[None], gt).max() == 0:
                out.append(cand)
                break
    return np.array(out, np.float64).reshape(-1, 4)


def token_centres(size=640, strides=(8, 16, 32)):
    cs = []
    for s in strides:
        n = size // s
        g = (np.arange(n) + .5) / n
        yy, xx = np.meshgrid(g, g, indexing='ij')
        cs.append(np.stack([xx.ravel(), yy.ravel()], 1))
    return np.concatenate(cs), (size // strides[0]) ** 2


class EvalImages:
    def __init__(self, folder, images):
        self.folder, self.images = folder, images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        import torchvision.transforms.functional as F
        from PIL import Image
        image = Image.open(f"{self.folder}/{self.images[index]['file_name']}").convert('RGB')
        return F.to_tensor(F.resize(image, [640, 640])), index


def extract(args):
    import torch
    from engine.core import YAMLConfig
    from error_gallery import ROOT
    chosen = set(np.load(args.d12 / 'similarity.npz')['train_images'].tolist())
    cfg = YAMLConfig(str(args.run / 'resolved_config.yml'))
    model = cfg.model
    model.load_state_dict(torch.load(args.run / 'best_stg2.pth', map_location='cpu', weights_only=False)['ema']['module'])
    model = model.to(args.device).eval()
    grab = {}
    model.backbone.register_forward_hook(lambda m, i, o: grab.update(bb=[t.detach() for t in o]))
    model.encoder.register_forward_hook(lambda m, i, o: grab.update(enc=[t.detach() for t in (o[0] if isinstance(o, tuple) else o)]))
    model.decoder.enc_score_head.register_forward_hook(lambda m, i, o: grab.update(enc_logits=o.detach()))
    centres, first_level = token_centres()
    centres_t = torch.as_tensor(centres, dtype=torch.float32, device=args.device)
    for split in ('train', 'val'):
        data = json.load(open(ROOT / f'Annotations/{split}.json'))
        by_image = defaultdict(list)
        for a in data['annotations']:
            by_image[a['image_id']].append(a)
        images = [i for i in data['images'] if split == 'val' or i['id'] in chosen]
        loader = torch.utils.data.DataLoader(EvalImages(ROOT / f'JPEGImages/{split}', images), batch_size=32, num_workers=8)
        rows = defaultdict(list)
        with torch.no_grad():
            for batch, indices in loader:
                batch = batch.to(args.device)
                model(batch)
                assert grab['enc_logits'].shape[1] == len(centres), 'encoder token grid differs from 640 / (8, 16, 32)'
                for k, index in enumerate(indices.tolist()):
                    image = images[index]
                    anns = by_image[image['id']]
                    if not anns:
                        continue
                    w, h = image['width'], image['height']
                    gt = np.array([[a['bbox'][0] / w, a['bbox'][1] / h, (a['bbox'][0] + a['bbox'][2]) / w, (a['bbox'][1] + a['bbox'][3]) / h] for a in anns], np.float64)
                    gt = np.clip(gt, 0, 1)
                    neg = negative_boxes(gt, np.random.default_rng(image['id']))
                    boxes = torch.as_tensor(np.concatenate([gt, neg]), dtype=torch.float32, device=args.device)
                    rows['IMG'].append(crops(batch[k], boxes).half().cpu().numpy())
                    rows['BB'].append(torch.cat([roi_mean(f[k], boxes) for f in grab['bb']], 1).half().cpu().numpy())
                    rows['ENC'].append(torch.cat([roi_mean(f[k], boxes) for f in grab['enc']], 1).half().cpu().numpy())
                    logit = grab['enc_logits'][k].float().max(1).values                                   # tokens
                    inside = ((centres_t[None, :, 0] >= boxes[:, None, 0]) & (centres_t[None, :, 0] <= boxes[:, None, 2]) &
                              (centres_t[None, :, 1] >= boxes[:, None, 1]) & (centres_t[None, :, 1] <= boxes[:, None, 3]))
                    best = torch.where(inside, logit[None], torch.full_like(logit[None], -1e9)).max(1).values
                    centre = (boxes[:, :2] + boxes[:, 2:]) / 2
                    nearest = ((centres_t[None, :first_level] - centre[:, None]) ** 2).sum(-1).argmin(1)
                    rows['ENC_SCORE'].append(torch.where(inside.any(1), best, logit[nearest]).cpu().numpy())
                    rows['positive'].append(np.r_[np.ones(len(gt), bool), np.zeros(len(neg), bool)])
                    rows['ann'].append(np.r_[[a['id'] for a in anns], np.full(len(neg), -1)])
                    rows['label'].append(np.r_[[a['category_id'] for a in anns], np.full(len(neg), -1)])
                    rows['image'].append(np.full(len(boxes), image['id']))
                    rows['size'].append(np.r_[[image['width']] * len(boxes)])
                    rows['long_side'].append(np.r_[[max(a['bbox'][2], a['bbox'][3]) for a in anns], np.full(len(neg), -1.)])
        np.savez(args.out / f'features_{split}.npz', **{k: np.concatenate(v) for k, v in rows.items()})
        print('written', args.out / f'features_{split}.npz', sum(len(v) for v in rows['ann']), flush=True)


def fit_probe(x, y, device, steps=400):
    """L2-regularised logistic regression on standardised features. Returns a scoring function."""
    import torch
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True) + 1e-6
    x = (x - mean) / std
    w = torch.zeros(x.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y) + 1e-3 * (w ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    w, b = w.detach(), b.detach()
    return lambda z: (((torch.as_tensor(z, dtype=torch.float32, device=device) - mean) / std) @ w + b).cpu().numpy()


def auc(pos, neg):
    order = np.argsort(np.r_[pos, neg], kind='stable')
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cell_name(size, name, long_side):
    s = str(int(size)) if int(size) in (224, 256, 512, 800, 1000) else 'misc'
    if name == 'tank' and s == '256':
        return '256/tank<16' if long_side < 16 else '256/tank>=16'
    return f'{s}/{name}'


def analyse(args):
    from error_gallery import ROOT
    names = {c['id']: c['name'] for c in json.load(open(ROOT / 'Annotations/val.json'))['categories']}
    sim = np.load(args.d12 / 'similarity.npz')
    sim_of = {**dict(zip(sim['val_ann'].tolist(), sim['val_sim'].tolist())), }
    sim_train = dict(zip(sim['train_ann'].tolist(), sim['train_sim'].tolist()))
    tr, va = (dict(np.load(args.out / f'features_{s}.npz')) for s in ('train', 'val'))
    ids = np.unique(tr['image'])
    perm = np.random.default_rng(0).permutation(ids)
    split_a = set(perm[:len(perm) * 2 // 3].tolist())
    in_a = np.array([i in split_a for i in tr['image']])
    pops = {
        'train B': dict(d=tr, m=~in_a, sim=np.array([sim_train.get(int(a), np.nan) for a in tr['ann']])),
        'val': dict(d=va, m=np.ones(len(va['ann']), bool), sim=np.array([sim_of.get(int(a), np.nan) for a in va['ann']])),
    }
    report = dict(counts=dict(train_a=dict(pos=int((in_a & tr['positive']).sum()), neg=int((in_a & ~tr['positive']).sum()))), features={})
    scores = {}
    for feat in FEATURES + ['ENC_SCORE']:
        if feat == 'ENC_SCORE':
            score = {k: p['d']['ENC_SCORE'].astype(np.float64) for k, p in pops.items()}
        else:
            probe = fit_probe(tr[feat][in_a].astype(np.float32), tr['positive'][in_a], args.device)
            score = {k: probe(p['d'][feat].astype(np.float32)) for k, p in pops.items()}
        scores[feat] = score
        val_neg = score['val'][~va['positive']]
        out = dict(threshold_fpr={})
        for fpr in (.01, .001):
            thr = float(np.quantile(val_neg, 1 - fpr))
            table = {}
            for k, p in pops.items():
                pos = p['m'] & p['d']['positive']
                neg = p['m'] & ~p['d']['positive']
                row = dict(negatives_fpr=r3((score[k][neg] > thr).mean() * 100))
                for label, lo, hi in SIM_BINS:
                    m = pos & (p['sim'] >= lo) & (p['sim'] < hi)
                    row[label] = dict(gt=int(m.sum()), recall=r3((score[k][m] > thr).mean() * 100), auc=r3(auc(score[k][m], score[k][neg])))
                m = pos & (p['sim'] < .9)
                row['novel (sim<.9)'] = dict(gt=int(m.sum()), recall=r3((score[k][m] > thr).mean() * 100), auc=r3(auc(score[k][m], score[k][neg])))
                # per cell, novel and seen
                cells = {}
                cell = np.array([cell_name(s, names[int(c)], ls) if c >= 0 else '' for s, c, ls in zip(p['d']['size'], p['d']['label'], p['d']['long_side'])])
                for c in np.unique(cell[pos]):
                    mc = pos & (cell == c)
                    if mc.sum() < 150:
                        continue
                    cells[c] = {lab: dict(gt=int((mc & sel).sum()), recall=r3((score[k][mc & sel] > thr).mean() * 100) if (mc & sel).sum() >= 20 else None)
                                for lab, sel in (('seen', p['sim'] >= .9), ('novel', p['sim'] < .9))}
                row['cells'] = cells
                table[k] = row
            out['threshold_fpr'][f'{fpr:g}'] = table
        report['features'][feat] = out
        t = out['threshold_fpr']['0.01']
        print(feat, {k: (t[k]['sim>=.9']['recall'], t[k]['novel (sim<.9)']['recall'], t[k]['negatives_fpr']) for k in pops}, flush=True)
    (args.out / 'generalisation_probe.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('written', args.out / 'generalisation_probe.json', flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract')
    e.add_argument('run', type=Path)
    e.add_argument('out', type=Path)
    e.add_argument('--d12', type=Path, required=True)
    e.add_argument('--device', default='cuda:0')
    a = sub.add_parser('analyse')
    a.add_argument('out', type=Path)
    a.add_argument('--d12', type=Path, required=True)
    a.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {'extract': extract, 'analyse': analyse}[args.command](args)


if __name__ == '__main__':
    main()
