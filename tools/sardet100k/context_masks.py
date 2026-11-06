"""Val-only, CPU: noise-free versions of the target/context masks. Imports helpers from detector_behaviour.py."""
import importlib.util
import json
import multiprocessing as mp
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

spec = importlib.util.spec_from_file_location('behaviour', '/tmp/detector_behaviour.py')
b = importlib.util.module_from_spec(spec)
sys.modules['behaviour'] = b
spec.loader.exec_module(b)


class Masked:
    def __init__(self, rows, mode):
        self.rows, self.mode = rows, mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torchvision.transforms.functional as F
        row = self.rows[index]
        values = np.asarray(Image.open(b.ROOT / 'JPEGImages/val' / row['file_name']).convert('L'), np.float32).copy()
        h, w = values.shape
        boxes = [g['box'] for g in b.gts[row['id']]]
        keep = np.zeros((h, w), bool)
        core = np.zeros((h, w), bool)
        for box in boxes:
            bw, bh = box[2] - box[0], box[3] - box[1]
            x0, y0 = max(int(box[0] - .25 * bw), 0), max(int(box[1] - .25 * bh), 0)
            x1, y1 = min(int(np.ceil(box[2] + .25 * bw)), w), min(int(np.ceil(box[3] + .25 * bh)), h)
            keep[y0:y1, x0:x1] = True
            core[max(int(box[1]), 0):min(int(np.ceil(box[3])), h), max(int(box[0]), 0):min(int(np.ceil(box[2])), w)] = True
        if self.mode == 'context_constant':
            outside = values[~keep]
            if outside.size:
                values[~keep] = np.median(outside)
        elif self.mode == 'context_blurred':
            values[~keep] = ndi.gaussian_filter(values, min(h, w) / 32.)[~keep]
        elif self.mode == 'target_inpainted':
            weights = (~core).astype(np.float32)
            sigma = max(np.median([max(box[2] - box[0], box[3] - box[1]) for box in boxes]) / 2., 2.)
            smooth = ndi.gaussian_filter(values * weights, sigma) / np.maximum(ndi.gaussian_filter(weights, sigma), 1e-3)
            values[core] = smooth[core]
        image = Image.fromarray(np.clip(values, 0, 255).astype(np.uint8)).convert('RGB')
        return F.to_tensor(F.resize(image, [640, 640])), index, w, h


def main():
    import torch
    from engine.core import YAMLConfig
    with mp.Pool(16) as pool:
        seen = b.seen_ids(pool)
    torch.set_num_threads(16)
    cfg = YAMLConfig(str(b.RUN / 'resolved_config.yml'))
    model, post = cfg.model, cfg.postprocessor
    model.load_state_dict(torch.load(b.RUN / 'best_stg2.pth', map_location='cpu')['ema']['module'])
    model, post = model.eval(), post.eval()
    rng = np.random.default_rng(0)
    chosen = []
    for c in b.names:
        ids = [i for i, rows in b.gts.items() if any(g['cls'] == c for g in rows)]
        chosen += rng.permutation(ids)[:150].tolist()
    selected = [b.images[i] for i in dict.fromkeys(chosen)]
    result = {}
    for mode in ('original', 'context_constant', 'context_blurred', 'target_inpainted'):
        loader = torch.utils.data.DataLoader(Masked(selected, mode), batch_size=16, num_workers=4)
        scores, fp_counts = {}, []
        with torch.no_grad():
            for batch, order, w, h in loader:
                for i, r in zip(order.tolist(), post(model(batch), torch.stack([w, h], 1))):
                    image_id = selected[i]['id']
                    pc, ps, pb = r['labels'].numpy().astype(int), r['scores'].numpy(), r['boxes'].numpy()
                    for g in b.gts[image_id]:
                        same = pc == g['cls']
                        m = b.iou(pb[same], g['box'][None])[:, 0] if same.any() else np.zeros(0)
                        ok = m >= .5
                        scores[g['id']] = float(ps[same][ok].max()) if ok.any() else 0.
                    boxes = np.stack([g['box'] for g in b.gts[image_id]])
                    high = ps >= .3
                    fp_counts.append(int((b.iou(pb[high], boxes).max(1) < .1).sum()) if high.any() else 0)
        result[mode] = dict(scores=scores, fp=round(float(np.mean(fp_counts)), 3))
        print('mode', mode, flush=True)
    meta = {g['id']: g for rows in b.gts.values() for g in rows}
    original = result['original']['scores']
    report = dict(original_fp=result['original']['fp'])
    for mode in ('context_constant', 'context_blurred', 'target_inpainted'):
        now = result[mode]['scores']
        entry = {}
        groups = {'all': lambda g: True, 'seen': lambda g: g['id'] in seen, 'unseen': lambda g: g['id'] not in seen, 'lt16': lambda g: g['side'] < 16, 'ge16': lambda g: g['side'] >= 16}
        for c, name in b.names.items():
            groups[name] = lambda g, c=c: g['cls'] == c
            groups[name + '_seen'] = lambda g, c=c: g['cls'] == c and g['id'] in seen
            groups[name + '_unseen'] = lambda g, c=c: g['cls'] == c and g['id'] not in seen
        for label, test in groups.items():
            ids = [i for i in original if test(meta[i]) and original[i] >= .3]
            if len(ids) >= 15:
                entry[label] = [len(ids), round(float(np.mean([now[i] >= .3 for i in ids])), 3), round(float(np.median([now[i] for i in ids])), 3)]
        report[mode] = dict(groups=entry, fp=result[mode]['fp'])
    (b.OUT / 'context_masks.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('RESULT', json.dumps(report, ensure_ascii=False), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
