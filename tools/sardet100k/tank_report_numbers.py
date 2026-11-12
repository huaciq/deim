"""CPU-only follow-up tables for docs/sardet100k_ogsod_tank_feature_diagnosis.md, read from the outputs of tank_feature_diagnosis.py.

Usage: python tools/sardet100k/tank_report_numbers.py OUT
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tank_feature_diagnosis import auc, cluster_ci, funnel_stage  # noqa: E402

OUT = Path(sys.argv[1])
rows = json.load(open(OUT / 'gt_rows.json'))
layout = {int(k): v for k, v in json.load(open(OUT / 'train_layout.json')).items()}
stages = json.load(open(OUT / 'stage_rows.json'))
native = {int(k): v for k, v in json.load(open(OUT / 'native_stats.json')).items()}
by_id = {r['gt_id']: r for r in rows}
report = {}
for r in rows:
    r['funnel'] = funnel_stage(r)
    info = layout.get(r['gt_id'])
    r['scene'] = 'na' if info is None else ('seen' if info['image_share'] >= .8 else 'unseen')
    r['hit'] = r['status'] == 'TP'
og = [r for r in rows if r['source'] == 'ogsod']
ms = [r for r in rows if r['source'] == 'msar']

report['scene_table'] = {}
for name, part in (('ogsod', og), ('msar', ms)):
    for scene in ('seen', 'unseen', 'na'):
        subset = [r for r in part if r['scene'] == scene]
        if subset:
            report['scene_table'][f'{name}|{scene}'] = dict(images=len({r['image_id'] for r in subset}), gt=len(subset), miss=sum(not r['hit'] for r in subset), recall=round(float(np.mean([r['hit'] for r in subset])), 3), funnel=dict(Counter(r['funnel'] for r in subset)), status=dict(Counter(r['status'] for r in subset)))
report['unseen_images'] = sorted({(r['file_name'], r['n_gt']) for r in og if r['scene'] != 'seen'})
report['zero_area'] = [dict(file=r['file_name'], source=r['source'], w=r['native_gw'], h=r['native_gh'], status=r['status']) for r in rows if r['native_gw'] <= 0 or r['native_gh'] <= 0]

# Size x crowding x scene, in OGSOD.
report['ogsod_size_scene'] = {}
for low, high in ((0, 16), (16, 24), (24, 1e5)):
    for scene in ('seen', 'unseen'):
        subset = [r for r in og if low <= r['input_side'] < high and r['scene'] == scene]
        report['ogsod_size_scene'][f'{low}-{high}|{scene}'] = dict(gt=len(subset), images=len({r['image_id'] for r in subset}), recall=round(float(np.mean([r['hit'] for r in subset])), 3) if subset else None, enc_prob=round(float(np.median([r['enc_prob_max_in_gt'] for r in subset])), 3) if subset else None)

# Size-stratified hit/miss separation inside seen scenes only: removes the scene effect and most of the size effect.
bins = ((0, 14), (14, 18), (18, 22), (22, 1e5))
by_stage = defaultdict(list)
for s in stages:
    if s['kind'] == 'gt':
        r = by_id[s['owner']]
        if r['source'] == 'ogsod' and r['scene'] == 'seen':
            by_stage[s['stage']].append(s)
report['seen_only_hit_vs_miss'] = {}
for stage, items in by_stage.items():
    flag = np.array([by_id[s['owner']]['hit'] for s in items])
    side = np.array([by_id[s['owner']]['input_side'] for s in items])
    images = [s['image_id'] for s in items]
    entry = {}
    for metric in ('ratio', 'z', 'peak_pct', 'cos_dist'):
        v = np.array([s.get(metric, np.nan) for s in items], float)
        if not np.isfinite(v).any():
            continue
        parts = []
        for a, b in bins:
            m = (side >= a) & (side < b)
            if (m & flag).sum() and (m & ~flag).sum():
                parts.append((auc(v[m & flag], v[m & ~flag]), min((m & flag).sum(), (m & ~flag).sum())))
        entry[metric] = dict(stratified=round(float(np.average([p[0] for p in parts], weights=[p[1] for p in parts])), 3), ci_plain=cluster_ci(v, flag, images, boot=200))
    report['seen_only_hit_vs_miss'][stage] = entry
fields = ('input_side', 'nn_input', 'enc_prob_max_in_gt', 'enc_rank_best_in_gt', 'enc_best_iou_selected', 'q_enc_tank_prob', 'q_score_0', 'q_score_3')
seen = [r for r in og if r['scene'] == 'seen']
flag, side = np.array([r['hit'] for r in seen]), np.array([r['input_side'] for r in seen])
report['seen_only_head'] = {}
for field in fields:
    v = np.array([r[field] for r in seen], float)
    parts = []
    for a, b in bins:
        m = (side >= a) & (side < b)
        if (m & flag).sum() and (m & ~flag).sum():
            parts.append((auc(v[m & flag], v[m & ~flag]), min((m & flag).sum(), (m & ~flag).sum())))
    report['seen_only_head'][field] = round(float(np.average([p[0] for p in parts], weights=[p[1] for p in parts])), 3)

# Encoder-to-final: how much the decoder changes the query that ends up being the best one.
report['decoder_change'] = {}
for name, test in (('ogsod_hit', lambda r: r['source'] == 'ogsod' and r['hit']), ('ogsod_LowScore', lambda r: r['source'] == 'ogsod' and r['status'] == 'LowScore'), ('ogsod_Loc', lambda r: r['source'] == 'ogsod' and r['status'] == 'Loc'), ('msar_hit', lambda r: r['source'] == 'msar' and r['hit'])):
    subset = [r for r in rows if test(r)]
    report['decoder_change'][name] = dict(n=len(subset), iou_gain=round(float(np.median([r['q_iou_3'] - r['q_init_iou'] for r in subset])), 3), score_gain=round(float(np.median([r['q_score_3'] - r['q_enc_tank_prob'] for r in subset])), 3),
                                          iou_gain_share_positive=round(float(np.mean([r['q_iou_3'] > r['q_init_iou'] for r in subset])), 3), final_vs_best_cell_iou=round(float(np.median([r['dec3_best_iou'] - r['enc_best_iou_all'] for r in subset])), 3))
ogsod_q = [r for r in og if r['native_gw'] > 0 and r['native_gh'] > 0]
report['enc_final_spearman'] = {}
for name, part in (('ogsod', ogsod_q), ('msar', [r for r in ms if r['native_gw'] > 0 and r['native_gh'] > 0])):
    a = np.argsort(np.argsort([r['q_enc_tank_prob'] for r in part]))
    b = np.argsort(np.argsort([r['q_score_3'] for r in part]))
    report['enc_final_spearman'][name] = round(float(np.corrcoef(a, b)[0, 1]), 3)

# The IoU a perfect detector reaches if labels are the true box rounded to integers, and with 1 px annotation noise.
rng = np.random.default_rng(0)


def ceiling(part, reach, offset=0.):
    w = np.array([r['native_gw'] for r in part])
    h = np.array([r['native_gh'] for r in part])
    keep = (w > 0) & (h > 0)
    w, h = w[keep], h[keep]
    values = []
    for _ in range(200):
        e = rng.uniform(-reach, reach, (4, len(w)))
        x1, y1, x2, y2 = e[0] + offset, e[1], w + e[2] + offset, h + e[3]  # True box relative to the label [0,w]x[0,h].
        iw = np.clip(np.minimum(x2, w) - np.maximum(x1, 0), 0, None)
        ih = np.clip(np.minimum(y2, h) - np.maximum(y1, 0), 0, None)
        inter = iw * ih
        values.append(inter / ((x2 - x1) * (y2 - y1) + w * h - inter))
    values = np.concatenate(values)
    return dict(median=round(float(np.median(values)), 3), ge_75=round(float((values >= .75).mean()), 3), ge_50=round(float((values >= .5).mean()), 3))


report['label_ceiling'] = {}
for name, part in (('ogsod', og), ('msar', ms)):
    report['label_ceiling'][name] = dict(rounding_half_px=ceiling(part, .5), noise_1px=ceiling(part, 1.), rounding_plus_x_offset_08=ceiling(part, .5, .8))
    tp = [r['tp_iou'] for r in part if r['hit']]
    report['label_ceiling'][name]['observed_tp'] = dict(median=round(float(np.median(tp)), 3), ge_75=round(float(np.mean(np.array(tp) >= .75)), 3))

# Native-pixel contrast across status.
report['native_by_status'] = {}
for name, test in (('hit', lambda r: r['hit']), ('NoBox', lambda r: r['status'] == 'NoBox'), ('LowScore', lambda r: r['status'] == 'LowScore'), ('Loc', lambda r: r['status'] == 'Loc')):
    part = [native[r['gt_id']] for r in og if test(r) and np.isfinite(native[r['gt_id']]['gt_mean'])]
    report['native_by_status'][name] = {k: round(float(np.nanmedian([p[k] for p in part])), 3) for k in part[0]}

# Status mix inside the 30 samples, with the scene flag.
samples = json.load(open(OUT / 'samples.json'))
report['samples_scene'] = dict(Counter(f"{s['group']}|{by_id[s['gt_id']]['scene']}" for s in samples))
report['sample_scene'] = {s['sample']: by_id[s['gt_id']]['scene'] for s in samples}
(OUT / 'report_numbers.json').write_text(json.dumps(report, indent=1, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))
print(json.dumps({k: report[k] for k in report if k not in ('sample_scene',)}, indent=0, ensure_ascii=False, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))
