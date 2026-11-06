"""Val-only, CPU, detector frozen. Does evidence computed from the target alone add ranking information beyond the score?

A small head sees only the box plus a 25% pad, cross-fitted over near-duplicate groups. Its logit is stacked with the
detector logit by a per-class logistic fit, with coefficients taken from the other fold.
"""
import importlib.util
import json
import multiprocessing as mp
import sys

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location('refine', '/tmp/refine_headroom.py')
r = importlib.util.module_from_spec(spec)
sys.modules['refine'] = r
spec.loader.exec_module(r)
SIDE = 32
MIN_SCORE = .05


def crops(image_id):
    im = r.images[image_id]
    values = np.asarray(Image.open(r.ROOT / 'JPEGImages/val' / im['file_name']).convert('L'), np.float32)
    H, W = values.shape
    pos = r.SLICES.get(image_id, np.array([], int))
    pos = pos[r.PS[pos] >= MIN_SCORE]
    rows = []
    for p in pos:
        x0, y0, x1, y1 = r.PB[p]
        w, h = max(x1 - x0, 1.), max(y1 - y0, 1.)
        px, py = max(.25 * w, 1.), max(.25 * h, 1.)
        ix0, iy0, ix1, iy1 = int(np.floor(x0 - px)), int(np.floor(y0 - py)), int(np.ceil(x1 + px)), int(np.ceil(y1 + py))
        canvas = np.zeros((iy1 - iy0, ix1 - ix0), np.float32)
        sx0, sy0, sx1, sy1 = max(ix0, 0), max(iy0, 0), min(ix1, W), min(iy1, H)
        if sx1 > sx0 and sy1 > sy0:
            canvas[sy0 - iy0:sy1 - iy0, sx0 - ix0:sx1 - ix0] = values[sy0:sy1, sx0:sx1]
        patch = np.asarray(Image.fromarray(canvas).resize((SIDE, SIDE), Image.BILINEAR), np.float32)
        rows.append((int(p), patch.astype(np.uint8), float(W)))
    return rows


def thumb(row):
    image = Image.open(r.ROOT / 'JPEGImages/val' / row['file_name']).convert('L')
    t = np.asarray(image.resize((32, 32), Image.BILINEAR), np.float32).ravel()
    t = t - t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def main():
    import torch
    import torch.nn as nn
    torch.set_num_threads(16)
    with mp.Pool(16) as pool:
        thumbs = np.stack(pool.map(thumb, r.val['images'], chunksize=64))
        parts = pool.map(crops, list(r.images), chunksize=32)
    parent = list(range(len(thumbs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for start in range(0, len(thumbs), 512):
        scores = thumbs[start:start + 512] @ thumbs.T
        for a, b in zip(*np.nonzero(scores >= .9)):
            parent[find(start + int(a))] = find(int(b))
    groups = np.array([find(i) for i in range(len(thumbs))])
    uniq = np.unique(groups)
    rng = np.random.default_rng(0)
    assign = dict(zip(uniq.tolist(), rng.integers(0, 2, len(uniq)).tolist()))
    fold_of = {i['id']: assign[int(g)] for i, g in zip(r.val['images'], groups)}
    rows = [x for part in parts for x in part]
    index = np.array([x[0] for x in rows])
    patches = torch.as_tensor(np.stack([x[1] for x in rows]), dtype=torch.float32)
    source_width = np.array([x[2] for x in rows], np.float32)
    fold = np.array([fold_of[int(r.PI[p])] for p in index])
    label = np.zeros(len(index), np.float32)
    per = {}
    for k, p in enumerate(index):
        per.setdefault(int(r.PI[p]), []).append(k)
    for image_id, members in per.items():
        members = np.array(members)
        for c in r.names:
            mine = members[r.PC[index[members]] == c]
            target = [g[1] for g in r.gts.get(image_id, []) if g[0] == c]
            if len(mine) and target:
                label[mine] = r.iou(r.PB[index[mine]], np.stack(target)).max(1) >= .5
    box = r.PB[index]
    w, h = np.maximum(box[:, 2] - box[:, 0], 1), np.maximum(box[:, 3] - box[:, 1], 1)
    score = np.clip(r.PS[index], 1e-4, 1 - 1e-4).astype(np.float64)
    logit = np.log(score / (1 - score))
    extras = torch.as_tensor(np.concatenate([np.eye(len(r.names), dtype=np.float32)[r.PC[index]], np.stack([np.log(w) / 5, np.log(h) / 5, np.log(source_width) / 7], 1)], 1), dtype=torch.float32)
    targets = torch.as_tensor(label)
    report = dict(detections=int(len(index)), positive_share=round(float(label.mean()), 3))

    def channels(batch):
        return torch.stack((batch / 255, torch.log1p(batch) / np.log(256), torch.log((batch + 1) / (batch.mean((1, 2), keepdim=True) + 1)) / 3), 1)

    class Head(nn.Module):
        def __init__(self, use_image):
            super().__init__()
            self.use_image = use_image
            layers, c = [], 3
            for out in (16, 32, 64, 96):
                layers += [nn.Conv2d(c, out, 3, padding=1, bias=False), nn.BatchNorm2d(out), nn.SiLU(), nn.MaxPool2d(2)]
                c = out
            self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.mlp = nn.Sequential(nn.Linear((96 if use_image else 0) + extras.shape[1], 128), nn.SiLU(), nn.Linear(128, 1))

        def forward(self, x, e):
            return self.mlp(torch.cat([self.net(x), e], 1) if self.use_image else e).squeeze(1)

    def crossfit(use_image):
        pred = np.zeros(len(index), np.float64)
        for test in (0, 1):
            tr = torch.as_tensor(np.flatnonzero(fold != test))
            te = torch.as_tensor(np.flatnonzero(fold == test))
            torch.manual_seed(0)
            model = Head(use_image)
            epochs = 10
            opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(tr) // 256))
            for _ in range(epochs):
                model.train()
                perm = tr[torch.randperm(len(tr))]
                for s in range(0, len(perm) - 255, 256):
                    b = perm[s:s + 256]
                    loss = nn.functional.binary_cross_entropy_with_logits(model(channels(patches[b]), extras[b]), targets[b])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sched.step()
            model.eval()
            with torch.no_grad():
                for s in range(0, len(te), 1024):
                    b = te[s:s + 1024]
                    pred[b.numpy()] = model(channels(patches[b]), extras[b]).numpy()
            print('fold', use_image, test, flush=True)
        return pred

    def stack_scores(evidence):
        result = r.PS.copy().astype(np.float32)
        coefficients = {}
        cls = r.PC[index]
        for test in (0, 1):
            for c, name in r.names.items():
                fit = (fold != test) & (cls == c)
                apply = (fold == test) & (cls == c)
                if not apply.any() or fit.sum() < 50 or label[fit].min() == label[fit].max():
                    continue
                X = torch.as_tensor(np.stack((logit[fit], evidence[fit]), 1), dtype=torch.float32)
                Y = torch.as_tensor(label[fit])
                wv = torch.tensor([1., 0.], requires_grad=True)
                bv = torch.zeros(1, requires_grad=True)
                opt = torch.optim.LBFGS([wv, bv], max_iter=100, line_search_fn='strong_wolfe')

                def closure():
                    opt.zero_grad()
                    loss = nn.functional.binary_cross_entropy_with_logits(X @ wv + bv, Y)
                    loss.backward()
                    return loss

                opt.step(closure)
                w_np, b_np = wv.detach().numpy(), bv.detach().numpy()
                result[index[apply]] = (1 / (1 + np.exp(-(w_np[0] * logit[apply] + w_np[1] * evidence[apply] + b_np[0])))).astype(np.float32)
                coefficients.setdefault(name, []).append([round(float(w_np[0]), 3), round(float(w_np[1]), 3)])
        return result, coefficients

    report['base'] = r.evaluate(r.PC, r.PS, r.PB, r.SLICES)
    for name, use_image in (('no_image', False), ('target_only', True)):
        evidence = crossfit(use_image)
        scores, coefficients = stack_scores(evidence)
        report[name] = dict(ap=r.evaluate(r.PC, scores, r.PB, r.SLICES), coefficients=coefficients)
        # Does stacking lift weak true targets and lower background false positives?
        new = scores[index]
        report[name]['positives_share_ge_0_3'] = [round(float((score[label == 1] >= .3).mean()), 3), round(float((new[label == 1] >= .3).mean()), 3)]
        report[name]['negatives_share_ge_0_3'] = [round(float((score[label == 0] >= .3).mean()), 3), round(float((new[label == 0] >= .3).mean()), 3)]
        print(name, report[name]['ap'].get('mean'), flush=True)
    (r.OUT.parent / 'behaviour/target_evidence.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('RESULT', json.dumps(report, ensure_ascii=False), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
