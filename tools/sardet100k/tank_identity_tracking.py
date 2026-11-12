"""D9: query identity evolution over training. For the same pair of competing queries of one tank GT, which comes
first: the classification score gap, or the change of the matched identity? Offline replay of the saved EMA
checkpoints of one run on the fixed val tank images of OGSOD and MSAR; no training, no model change.

Identity across checkpoints
  Decoder query j is built from the j-th token of the encoder Top-300, and that token is a fixed location of the
  flattened encoder memory (level and grid cell). Query indices are not comparable across checkpoints, token indices
  are. A query is therefore identified by its token; at a checkpoint where the token is outside the Top-300 the query
  does not exist (no decoder output, cannot be matched), but its encoder class logit is still defined.

Pairs (defined once, at the last checkpoint, last decoder layer)
  Candidates of a tank GT: queries whose highest-IoU tank GT is this GT, IoU >= .1 (survey rule).
  A = highest tank score (the final winner); B = highest-IoU candidate other than A, IoU >= .5.
  wrong: IoU(B) >= IoU(A) + .05 (the winner is the worse box; OGSOD tank's "480 pairs" of the tank diagnosis)
  right: IoU(A) >= IoU(B) + .05; tie: otherwise.

Per checkpoint, pair and stage (enc, pre, dec1-4), with the run's matcher on all GTs of the image (as in training):
  matched identity of the GT (A / B / other / none); tank logit, IoU and the weighted matching cost terms of A and B
  (class = focal cost x cost_class, L1 x cost_bbox, GIoU x cost_giou); MAL target and weight of A and B on the tank
  channel (matched: IoU^gamma and 1; otherwise 0 and p^gamma); whether the class term decided the pairwise preference
  (box cost prefers one, total cost the other); at dec4 also which candidate has the top score and the best IoU.

What it cannot tell: the replay uses EMA weights on val images without augmentation, not the online weights on
training batches; checkpoints are 4 epochs apart, so a score change and a matching change inside the same interval
cannot be ordered, and inside one checkpoint the matcher reads the current score by construction.

Training images: the same replay on train images (--split train) shows the matching state on images the model was
supervised on, still with EMA weights and without augmentation. The source of a train image is assigned by tank size
(see tank_images), not by pixel matching.

Commands:
  collect RUN OUT --checkpoint PATH [--threads 6] [--split val|train] [--max-msar 400]
                                                        (server, CPU) OUT/replay_<epoch>.pt
  analyse OUT [--out DIR]                               OUT/identity_tracking.json, fig_identity_tracking.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_report import iou_xyxy, r3, sigmoid  # noqa: E402

STAGES = ['enc', 'pre', 'dec1', 'dec2', 'dec3', 'dec4']
TANK = 3


# ---------------------------------------------------------------- collect (server)
def tank_images(split, out, max_msar, seed=0):
    """OGSOD and MSAR tank images of a split with their source.
    val: the pixel-matched source map (error_gallery_val/source.json).
    train: no source map exists on the server and the OGSOD/MSAR reference images are not there, so the source is
    assigned by the median long side of the image's tank GTs in a 256x256 image: < 16 px -> ogsod, >= 18 px -> msar,
    16-18 px left out. On val this rule agrees with the pixel match on 139 of 140 tank images (one OGSOD image with
    large tanks would be called MSAR). Every OGSOD-like image is kept; MSAR-like images are subsampled to max_msar."""
    from collections import defaultdict
    from error_gallery import ROOT
    from tank_feature_diagnosis import load_val
    if split == 'val':
        annotations, source, grouped = load_val(out)
        return annotations, grouped, [(i, source[i['id']]) for i in annotations['images']
                                      if source[i['id']] in ('ogsod', 'msar') and any(a['category_id'] == TANK for a in grouped[i['id']])]
    annotations = json.load(open(ROOT / f'Annotations/{split}.json'))
    grouped = defaultdict(list)
    for a in annotations['annotations']:
        grouped[a['image_id']].append(a)
    chosen = {'ogsod': [], 'msar': []}
    for i in annotations['images']:
        sizes = [max(a['bbox'][2], a['bbox'][3]) for a in grouped[i['id']] if a['category_id'] == TANK]
        if not sizes or (i['width'], i['height']) != (256, 256):
            continue
        med = float(np.median(sizes))
        if med < 16:
            chosen['ogsod'].append(i)
        elif med >= 18:
            chosen['msar'].append(i)
    rng = np.random.default_rng(seed)
    msar = [chosen['msar'][k] for k in sorted(rng.choice(len(chosen['msar']), min(max_msar, len(chosen['msar'])), replace=False))]
    return annotations, grouped, [(i, 'ogsod') for i in chosen['ogsod']] + [(i, 'msar') for i in msar]


def collect(args):
    import torch
    from error_gallery import ROOT
    from tank_feature_diagnosis import to_input
    from tank_matching_counterfactual import Runner
    torch.set_num_threads(args.threads)
    annotations, grouped, selected = tank_images(args.split, args.out, args.max_msar)
    runner = Runner(args.run, 'cpu', args.checkpoint)
    m = runner.matcher
    images = []
    source = {i['id']: s for i, s in selected}
    for image, _ in selected:
        anns = grouped[image['id']]
        x, _ = to_input(ROOT / 'JPEGImages' / args.split / image['file_name'])
        targets, xy = runner.targets(image, anns)
        outs = runner.stages(x[None])
        ind = {s: runner.matcher({'pred_logits': outs[s][0][None], 'pred_boxes': outs[s][1][None]}, targets)['indices'][0] for s in STAGES}
        images.append(dict(
            image=image['id'], source=source[image['id']], labels=np.array([a['category_id'] for a in anns]), gt_xy=xy,
            gt_cxcywh=targets[0]['boxes'].numpy(), topk=runner.saved['topk_ind'][0].numpy().astype(np.int64),
            enc_tank_all=runner.saved['enc_logits'][0][:, TANK].float().numpy(),
            logits={s: outs[s][0][:, TANK].float().numpy() for s in STAGES},
            boxes={s: outs[s][1].float().numpy() for s in STAGES},
            matched={s: (ind[s][0].numpy(), ind[s][1].numpy()) for s in STAGES}))
    meta = dict(run=str(args.run), checkpoint=str(args.checkpoint), epoch=runner.epoch, images=len(images), split=args.split,
                sources={s: sum(v == s for v in source.values()) for s in ('ogsod', 'msar')},
                matcher=dict(cost_class=float(m.cost_class), cost_bbox=float(m.cost_bbox), cost_giou=float(m.cost_giou),
                             alpha=float(m.alpha), gamma=float(m.gamma)),
                mal_gamma=float(runner.criterion.gamma))
    torch.save(dict(meta=meta, images=images), args.out / f'replay_{runner.epoch:02d}.pt')
    print('written', args.out / f'replay_{runner.epoch:02d}.pt', len(images), flush=True)


# ---------------------------------------------------------------- analyse (local)
def xyxy(c):
    return np.stack([c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2, c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2], 1)


def giou(a, b):
    """a (N,4), b (4,) xyxy -> (N,)."""
    iou = iou_xyxy(a, b[None])[:, 0]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    inter = iou * (area_a + area_b) / (1 + iou)
    union = area_a + area_b - inter
    hull = (np.maximum(a[:, 2], b[2]) - np.minimum(a[:, 0], b[0])) * (np.maximum(a[:, 3], b[3]) - np.minimum(a[:, 1], b[1]))
    return iou - (hull - union) / np.maximum(hull, 1e-12)


def costs(meta, logit, box_c, gt_c):
    mc = meta['matcher']
    p = sigmoid(logit.astype(np.float64))
    neg = (1 - mc['alpha']) * p ** mc['gamma'] * -np.log(1 - p + 1e-8)
    pos = mc['alpha'] * (1 - p) ** mc['gamma'] * -np.log(p + 1e-8)
    cls = mc['cost_class'] * (pos - neg)
    l1 = mc['cost_bbox'] * np.abs(box_c - gt_c[None]).sum(1)
    gi = mc['cost_giou'] * -giou(xyxy(box_c), xyxy(gt_c[None])[0])
    return cls, l1, gi


def gt_candidates(img, stage, j_tank):
    """Candidate query indices of the j-th tank GT at a stage, and the IoU of every query with that GT."""
    tank = np.flatnonzero(img['labels'] == TANK)
    iou = iou_xyxy(xyxy(img['boxes'][stage].astype(np.float64)), img['gt_xy'][tank].astype(np.float64))
    owner, best = iou.argmax(1), iou.max(1)
    return np.flatnonzero((owner == j_tank) & (best >= .1)), iou[:, j_tank]


def define_pairs(final):
    pairs = []
    for img in final['images']:
        tank = np.flatnonzero(img['labels'] == TANK)
        for j, t in enumerate(tank):
            cand, iou = gt_candidates(img, 'dec4', j)
            if len(cand) < 2:
                continue
            a = int(cand[np.argmax(img['logits']['dec4'][cand])])
            rest = [q for q in cand if q != a and iou[q] >= .5]
            if not rest:
                continue
            b = int(max(rest, key=lambda q: iou[q]))
            kind = 'wrong' if iou[b] >= iou[a] + .05 else 'right' if iou[a] >= iou[b] + .05 else 'tie'
            pairs.append(dict(image=img['image'], source=img['source'], gt=int(t), j=j, kind=kind,
                              token_a=int(img['topk'][a]), token_b=int(img['topk'][b]), iou_a=float(iou[a]), iou_b=float(iou[b])))
    return pairs


def track(pair, replay):
    """Per-stage state of one pair at one checkpoint."""
    meta = replay['meta']
    img = replay['by_image'][pair['image']]
    pos = {int(tok): q for q, tok in enumerate(img['topk'])}
    qa, qb = pos.get(pair['token_a'], -1), pos.get(pair['token_b'], -1)
    gt_c = img['gt_cxcywh'][pair['gt']].astype(np.float64)
    g = meta['mal_gamma']
    out = dict(present_a=qa >= 0, present_b=qb >= 0,
               enc_all_diff=float(img['enc_tank_all'][pair['token_a']] - img['enc_tank_all'][pair['token_b']]))
    for s in STAGES:
        q_of = dict(zip(img['matched'][s][1].tolist(), img['matched'][s][0].tolist()))
        mq = q_of.get(pair['gt'], -1)
        out[f'match_{s}'] = 'none' if mq < 0 else 'A' if mq == qa else 'B' if mq == qb else 'other'
        if qa < 0 or qb < 0:
            continue
        q = np.array([qa, qb])
        logit = img['logits'][s][q]
        box = img['boxes'][s][q].astype(np.float64)
        iou = iou_xyxy(xyxy(box), xyxy(gt_c[None]))[:, 0]
        cls, l1, gi = costs(meta, logit, box, gt_c)
        p = sigmoid(logit.astype(np.float64))
        target = np.array([iou[k] ** g if q[k] == mq else 0. for k in range(2)])
        weight = np.array([1. if q[k] == mq else p[k] ** g for k in range(2)])
        box_cost, total = l1 + gi, cls + l1 + gi
        out[s] = dict(logit_diff=float(logit[0] - logit[1]), iou_diff=float(iou[0] - iou[1]), iou_a=float(iou[0]), iou_b=float(iou[1]),
                      cls_cost_diff=float(cls[1] - cls[0]), l1_cost_diff=float(l1[1] - l1[0]), giou_cost_diff=float(gi[1] - gi[0]),
                      mal_target=target.tolist(), mal_weight=weight.tolist(),
                      class_decides=bool(np.sign(box_cost[1] - box_cost[0]) != np.sign(total[1] - total[0])),
                      box_prefers='A' if box_cost[0] < box_cost[1] else 'B')
    cand, iou = gt_candidates(img, 'dec4', pair['j'])
    if len(cand):
        top, best = int(cand[np.argmax(img['logits']['dec4'][cand])]), int(cand[np.argmax(iou[cand])])
        out['top'] = 'A' if top == qa else 'B' if top == qb else 'other'
        out['best'] = 'A' if best == qa else 'B' if best == qb else 'other'
    return out


def lock_epoch(flags):
    """Index of the first checkpoint from which every flag is True up to the end; None if the last one is False."""
    if not flags[-1]:
        return None
    k = len(flags) - 1
    while k > 0 and flags[k - 1]:
        k -= 1
    return k


def ordering(states, epochs):
    """Score lead (A ahead at dec4; B absent counts as A ahead, A absent as not) vs matched identity (A at dec4)."""
    score = []
    for st in states:
        if 'dec4' in st:
            score.append(st['dec4']['logit_diff'] > 0)
        else:
            score.append(bool(st['present_a'] and not st['present_b']))
    match = [st['match_dec4'] == 'A' for st in states]
    ls, lm = lock_epoch(score), lock_epoch(match)
    order = ('never' if ls is None or lm is None else 'score_first' if ls < lm else 'match_first' if lm < ls else 'same_checkpoint')
    # transitions between consecutive checkpoints
    t_match = [score[k - 1] for k in range(1, len(states)) if not match[k - 1] and match[k]]
    t_score = [match[k - 1] for k in range(1, len(states)) if not score[k - 1] and score[k]]
    t_match_lost = [score[k - 1] for k in range(1, len(states)) if match[k - 1] and not match[k]]
    return dict(score=score, match=match, lock_score=None if ls is None else epochs[ls], lock_match=None if lm is None else epochs[lm],
                order=order, match_gain_after_score_ahead=t_match, score_gain_after_matched=t_score, match_loss_score_ahead=t_match_lost)


def analyse(args):
    import torch
    files = sorted(args.out.glob('replay_*.pt'))
    replays = [torch.load(f, weights_only=False) for f in files]
    for r in replays:
        r['by_image'] = {i['image']: i for i in r['images']}
    replays.sort(key=lambda r: r['meta']['epoch'])
    epochs = [r['meta']['epoch'] for r in replays]
    pairs = define_pairs(replays[-1])
    rows = []
    for p in pairs:
        states = [track(p, r) for r in replays]
        rows.append(dict(p, states=states, ordering=ordering(states, epochs)))
    groups = {}
    for src in ('ogsod', 'msar'):
        for kind in ('wrong', 'right', 'tie'):
            g = [r for r in rows if r['source'] == src and r['kind'] == kind]
            if g:
                groups[f'{src}/{kind}'] = g
    report = dict(epochs=epochs, meta=replays[-1]['meta'], pairs={k: len(v) for k, v in groups.items()}, groups={})
    rng = np.random.default_rng(0)
    for name, g in groups.items():
        per_epoch = []
        for e in range(len(epochs)):
            st = [r['states'][e] for r in g]
            both = [x for x in st if 'dec4' in x]
            row = dict(epoch=epochs[e], present_a=r3(np.mean([x['present_a'] for x in st])), present_b=r3(np.mean([x['present_b'] for x in st])),
                       both_present=len(both), enc_all_logit_diff_median=r3(np.median([x['enc_all_diff'] for x in st])))
            for s in STAGES:
                lab = [x[f'match_{s}'] for x in st]
                row[f'match_{s}'] = {k: r3(np.mean([v == k for v in lab])) for k in ('A', 'B', 'other', 'none')}
            for s in ('enc', 'dec1', 'dec2', 'dec4'):
                b2 = [x[s] for x in st if s in x]
                if not b2:
                    continue
                med = lambda k: r3(np.median([x[k] for x in b2]))  # noqa: E731
                row[s] = dict(logit_diff=med('logit_diff'), iou_diff=med('iou_diff'), iou_a=med('iou_a'), iou_b=med('iou_b'),
                              cls_cost_diff=med('cls_cost_diff'), l1_cost_diff=med('l1_cost_diff'), giou_cost_diff=med('giou_cost_diff'),
                              a_logit_ahead=r3(np.mean([x['logit_diff'] > 0 for x in b2])),
                              class_decides=r3(np.mean([x['class_decides'] for x in b2])),
                              box_prefers_b=r3(np.mean([x['box_prefers'] == 'B' for x in b2])),
                              mal_target_a=r3(np.median([x['mal_target'][0] for x in b2])), mal_target_b=r3(np.median([x['mal_target'][1] for x in b2])),
                              mal_weight_a=r3(np.median([x['mal_weight'][0] for x in b2])), mal_weight_b=r3(np.median([x['mal_weight'][1] for x in b2])),
                              b_positive=r3(np.mean([x['mal_target'][1] > 0 for x in b2])))
            tops = [x for x in st if 'top' in x]
            row['top_is'] = {k: r3(np.mean([x['top'] == k for x in tops])) for k in ('A', 'B', 'other')} if tops else None
            row['best_is'] = {k: r3(np.mean([x['best'] == k for x in tops])) for k in ('A', 'B', 'other')} if tops else None
            per_epoch.append(row)
        orders = [r['ordering'] for r in g]
        imgs = np.array([r['image'] for r in g])
        uniq = np.unique(imgs)
        by = {i: np.flatnonzero(imgs == i) for i in uniq}
        cats = ['score_first', 'match_first', 'same_checkpoint', 'never']
        share = {c: np.array([o['order'] == c for o in orders], float) for c in cats}
        boot = {c: [] for c in cats}
        for _ in range(1000):
            idx = np.concatenate([by[i] for i in rng.choice(uniq, len(uniq))])
            for c in cats:
                boot[c].append(share[c][idx].mean())
        flat = lambda k: [v for o in orders for v in o[k]]  # noqa: E731
        trans = {k: dict(events=len(flat(k)), share_true=r3(np.mean(flat(k))) if flat(k) else None)
                 for k in ('match_gain_after_score_ahead', 'score_gain_after_matched', 'match_loss_score_ahead')}
        base_score = r3(np.mean([v for o in orders for v in o['score'][:-1]]))
        base_match = r3(np.mean([v for o in orders for v in o['match'][:-1]]))
        report['groups'][name] = dict(
            pairs=len(g), images=len(uniq), per_epoch=per_epoch,
            order={c: dict(share=r3(share[c].mean()), ci95=[r3(np.percentile(boot[c], 2.5)), r3(np.percentile(boot[c], 97.5))]) for c in cats},
            lock_epoch_median=dict(score=r3(np.median([o['lock_score'] for o in orders if o['lock_score'] is not None])) if any(o['lock_score'] is not None for o in orders) else None,
                                   match=r3(np.median([o['lock_match'] for o in orders if o['lock_match'] is not None])) if any(o['lock_match'] is not None for o in orders) else None),
            transitions=trans, base_rate=dict(score_ahead=base_score, matched=base_match))
        print(name, len(g), {c: report['groups'][name]['order'][c]['share'] for c in cats}, trans, flush=True)
    dest = args.dest or args.out
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'identity_tracking.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))
    torch.save(rows, dest / 'identity_tracking_rows.pt')
    plot(report, dest)
    print('written', dest / 'identity_tracking.json', flush=True)


def plot(report, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [n for n in ('ogsod/wrong', 'ogsod/right', 'msar/right', 'msar/tie') if n in report['groups']]
    fig, axes = plt.subplots(4, len(names), figsize=(4.2 * len(names), 12), squeeze=False, sharex=True)
    for c, n in enumerate(names):
        pe = report['groups'][n]['per_epoch']
        ep = [r['epoch'] for r in pe]
        ax = axes[0, c]
        for s, col in (('enc', '#9aa5b1'), ('dec1', '#5b8def'), ('dec4', '#1d3f8f')):
            ax.plot(ep, [r[s]['logit_diff'] if s in r else np.nan for r in pe], marker='o', color=col, label=s)
        ax.axhline(0, color='k', lw=.6)
        ax.set_title(f'{n} ({report["groups"][n]["pairs"]} pairs)')
        ax.set_ylabel('logit A - B (median)') if c == 0 else None
        ax.legend(fontsize=8)
        ax = axes[1, c]
        ax.plot(ep, [r['dec4']['iou_diff'] if 'dec4' in r else np.nan for r in pe], marker='o', color='#c0392b', label='IoU A - B (dec4)')
        ax.plot(ep, [r['enc']['iou_diff'] if 'enc' in r else np.nan for r in pe], marker='o', color='#e6a39a', label='IoU A - B (enc)')
        ax.axhline(0, color='k', lw=.6)
        ax.set_ylabel('IoU A - B (median)') if c == 0 else None
        ax.legend(fontsize=8)
        ax = axes[2, c]
        for k, col in (('cls_cost_diff', '#1d3f8f'), ('giou_cost_diff', '#c0392b'), ('l1_cost_diff', '#e67e22')):
            ax.plot(ep, [r['dec4'][k] if 'dec4' in r else np.nan for r in pe], marker='o', color=col, label=k.replace('_cost_diff', '') + ' cost B - A')
        ax.axhline(0, color='k', lw=.6)
        ax.set_ylabel('cost B - A, dec4 (median)') if c == 0 else None
        ax.legend(fontsize=8)
        ax = axes[3, c]
        bottom = np.zeros(len(ep))
        for k, col in (('A', '#1d3f8f'), ('B', '#c0392b'), ('other', '#bbbbbb'), ('none', '#eeeeee')):
            v = np.array([r['match_dec4'][k] for r in pe])
            ax.bar(ep, v, bottom=bottom, width=2.6, color=col, label=f'dec4 matched {k}')
            bottom += v
        ax.plot(ep, [r['match_enc']['A'] for r in pe], color='#f1c40f', marker='s', label='enc matched A')
        ax.plot(ep, [r['top_is']['A'] if r['top_is'] else np.nan for r in pe], color='k', ls='--', label='top score is A')
        ax.set_ylim(0, 1)
        ax.set_xlabel('epoch')
        ax.set_ylabel('share of pairs') if c == 0 else None
        ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(dest / 'fig_identity_tracking.png', dpi=110)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('run', type=Path)
    c.add_argument('out', type=Path)
    c.add_argument('--checkpoint', type=Path, required=True)
    c.add_argument('--threads', type=int, default=6)
    c.add_argument('--split', choices=['val', 'train'], default='val')
    c.add_argument('--max-msar', type=int, default=400, help='train only: MSAR-like images kept (random, seed 0)')
    a = sub.add_parser('analyse')
    a.add_argument('out', type=Path)
    a.add_argument('--dest', type=Path)
    args = parser.parse_args()
    if args.command == 'collect':
        args.out.mkdir(parents=True, exist_ok=True)
    {'collect': collect, 'analyse': analyse}[args.command](args)


if __name__ == '__main__':
    main()
