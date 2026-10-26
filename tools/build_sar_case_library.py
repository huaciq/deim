#!/usr/bin/env python3
"""Build measured SAR hard-case candidates with context and enlarged crops.

Needs only numpy/Pillow. Feature tags are observable proxies, not physical
ground-truth labels. Outputs stay separate from annotations and checkpoints.
"""
import argparse
from collections import defaultdict
import csv
import html
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.sar_analysis_core import error_analysis, load_json, save_json

GROUPS = dict(clutter='背景纹理/杂波候选', low_contrast='低对比度候选',
              dense='密集目标', appearance='外观稀疏或细长候选', miss='漏检',
              background='背景误检', localization='定位错误', duplicate='重复检测',
              classification='分类错误', classification_localization='分类与定位同时错误')
COLORS = dict(tp='#39c9f0', background='#ff4545', localization='#ffa536', duplicate='#ffff33',
              classification='#ff55df', classification_localization='#ae7eff', miss='#ff4545')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def regional_features(gray, box, all_boxes):
    height, width = gray.shape
    x, y, w, h = box
    left, top = max(0, int(x - .5 * w)), max(0, int(y - .5 * h))
    right, bottom = min(width, int(np.ceil(x + 1.5 * w))), min(height, int(np.ceil(y + 1.5 * h)))
    patch = gray[top:bottom, left:right]
    yy, xx = np.mgrid[top:bottom, left:right]
    foreground = (xx >= x) & (xx < x + w) & (yy >= y) & (yy < y + h)
    background = ~foreground
    for gx, gy, gw, gh in all_boxes:
        background &= ~((xx >= gx) & (xx < gx + gw) & (yy >= gy) & (yy < gy + gh))
    fg = patch[foreground]
    bg = patch[background]
    if len(fg) == 0 or len(bg) < 16:
        return dict(cnr=None, background_cv=None, texture=None, bright_fraction=None)
    fg_mean, bg_mean, bg_std = float(fg.mean()), float(bg.mean()), float(bg.std())
    bright_fraction = float((fg > bg_mean + 2 * max(bg_std, 1)).mean())
    dx = np.abs(patch[:, 1:] - patch[:, :-1])
    valid = background[:, 1:] & background[:, :-1]
    return dict(cnr=abs(fg_mean - bg_mean) / max(bg_std, 1),
                background_cv=bg_std / max(bg_mean, 1),
                texture=float(dx[valid].mean()) if valid.any() else 0.0,
                bright_fraction=bright_fraction)


def crop_rectangle(box, image_size):
    width, height = image_size
    x, y, w, h = box
    side = min(max(2.7 * max(w, h), 150), max(width, height))
    cx, cy = x + w / 2, y + h / 2
    left, top = max(0, min(width - side, cx - side / 2)), max(0, min(height - side, cy - side / 2))
    return [int(left), int(top), int(min(width, left + side)), int(min(height, top + side))]


def load_font(size):
    for path in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf']:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotated_view(image, crop, ground_truth, records, names, focus_id, focus_prediction):
    view = image.crop(crop).convert('RGB').resize((620, 620))
    draw = ImageDraw.Draw(view)
    font = load_font(13)
    sx, sy = 620 / (crop[2] - crop[0]), 620 / (crop[3] - crop[1])
    def draw_box(box, color, label, weight=2):
        x, y, w, h = box
        xy = [(x - crop[0]) * sx, (y - crop[1]) * sy,
              (x + w - crop[0]) * sx, (y + h - crop[1]) * sy]
        if xy[2] < 0 or xy[3] < 0 or xy[0] > 620 or xy[1] > 620:
            return
        draw.rectangle(xy, outline=color, width=weight)
        tx, ty = max(0, xy[0]), max(0, xy[1] - 16)
        bounds = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bounds, fill='#10151d')
        draw.text((tx, ty), label, font=font, fill=color)
    for annotation in ground_truth:
        draw_box(annotation['bbox'], '#4bff62', 'GT ' + names[annotation['category_id']],
                 4 if annotation['id'] == focus_id else 2)
    for record in records:
        draw_box(record['bbox'], COLORS[record['kind']],
                 record['kind'] + ' ' + names[record['category_id']] + f" {record['score']:.2f}",
                 4 if record['prediction_index'] == focus_prediction else 2)
    return view


def make_case(image_path, item, ground_truth, records, names, destination):
    image = Image.open(image_path).convert('RGB')
    crop = crop_rectangle(item['bbox'], image.size)
    scene = annotated_view(image, [0, 0, *image.size], ground_truth, records, names,
                           item.get('gt_id'), item.get('prediction_index'))
    detail = annotated_view(image, crop, ground_truth, records, names,
                            item.get('gt_id'), item.get('prediction_index'))
    canvas = Image.new('RGB', (1280, 790), '#10151d')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), f"{item['case_id']} | {item['group']} | image={item['image_id']} | {item['status']}",
              font=load_font(21), fill='white')
    draw.text((20, 49), 'Original-image context', font=load_font(17), fill='#d8e1ec')
    draw.text((660, 49), 'Enlarged crop (display resize only)', font=load_font(17), fill='#d8e1ec')
    canvas.paste(scene, (20, 77))
    canvas.paste(detail, (660, 77))
    draw.text((20, 709), 'GT=green  TP=cyan  BG=red  Loc=orange  Dup=yellow  Class=pink  Class+Loc=purple',
              font=load_font(16), fill='#d8e1ec')
    measure = '  '.join(f'{k}={item[k]:.3f}' for k in ['cnr', 'background_cv', 'texture', 'bright_fraction'] if item.get(k) is not None)
    draw.text((20, 738), measure, font=load_font(16), fill='#d8e1ec')
    canvas.save(destination, quality=88)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-group', type=int, default=4)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_json(args.annotations)
    predictions = load_json(args.predictions)
    errors, records, misses, objects = error_analysis(dataset, predictions)
    names = {c['id']: c['name'] for c in dataset['categories']}
    images = {im['id']: im for im in dataset['images']}
    gts, dts = defaultdict(list), defaultdict(list)
    for annotation in dataset['annotations']:
        gts[annotation['image_id']].append(annotation)
    for record in records:
        dts[record['image_id']].append(record)
    object_map = {o['gt_id']: o for o in objects}
    features = []
    for j, image in enumerate(dataset['images']):
        gray = np.asarray(Image.open(args.image_root / image['file_name']).convert('L'), dtype=np.float32)
        targets = gts[image['id']]
        centers = np.asarray([[g['bbox'][0] + g['bbox'][2] / 2, g['bbox'][1] + g['bbox'][3] / 2] for g in targets])
        for i, target in enumerate(targets):
            box = target['bbox']
            size = np.sqrt(box[2] * box[3])
            distance = np.sqrt(((centers - centers[i]) ** 2).sum(axis=1)) / max(size, 1)
            distance[i] = np.inf
            features.append(dict(gt_id=target['id'], image_id=image['id'], category_id=target['category_id'],
                                 category=names[target['category_id']], bbox=box,
                                 detected=object_map[target['id']]['detected'], input_side=object_map[target['id']]['input_side'],
                                 gt_per_image=len(targets), close_neighbors=int((distance < 2).sum()),
                                 nearest_normalized=float(distance.min()) if len(targets) > 1 else None,
                                 aspect=max(box[2] / box[3], box[3] / box[2]),
                                 **regional_features(gray, box, [g['bbox'] for g in targets])))
        if j % 100 == 0:
            print(f'pixel features {j}/{len(images)}', flush=True)
    write_csv(output / 'object_appearance_features.csv', features)
    valid = [f for f in features if f['cnr'] is not None]
    ranked = dict(clutter=sorted(valid, key=lambda f:(f['detected'], -(f['background_cv'] + f['texture'] / 30))),
                  low_contrast=sorted(valid, key=lambda f:(f['detected'], f['cnr'])),
                  dense=sorted(valid, key=lambda f:(f['detected'], -f['close_neighbors'], -f['gt_per_image'])),
                  appearance=sorted([f for f in valid if f['cnr'] > .25],
                                    key=lambda f:(f['detected'], f['bright_fraction'] / f['aspect'])))
    candidates = []
    for group, values in ranked.items():
        used_images = set()
        selected = []
        # Include one successful detection as a difficult-scene comparison.
        successes = [f for f in values if f['detected']]
        failures = [f for f in values if not f['detected']]
        for feature in failures + successes:
            if feature['image_id'] not in used_images:
                selected.append(feature)
                used_images.add(feature['image_id'])
            if len(selected) >= args.per_group - (1 if successes else 0):
                break
        for feature in successes:
            if feature['image_id'] not in used_images and len(selected) < args.per_group:
                selected.append(feature)
                break
        candidates.extend(dict(f, group=group, status='TP comparison' if f['detected'] else 'FN',
                               prediction_index=None) for f in selected)
    error_candidates = dict(miss=misses, **{name:[r for r in records if r['kind'] == name]
                                          for name in GROUPS if name not in ranked and name != 'miss'})
    feature_map = {f['gt_id']: f for f in features}
    for group, values in error_candidates.items():
        used_images = set()
        for record in sorted(values, key=lambda r:-r.get('score', 0)):
            if record['image_id'] in used_images:
                continue
            feature = feature_map.get(record.get('gt_id'), {})
            candidates.append({**feature, **record, 'group':group, 'status':group,
                               'bbox':record['bbox']})
            used_images.add(record['image_id'])
            if len(used_images) >= args.per_group:
                break
    manifest = []
    for i, candidate in enumerate(candidates, 1):
        image = images[candidate['image_id']]
        candidate.update(case_id=f'case_{i:03d}', file_name=image['file_name'])
        name = f"{candidate['case_id']}_{candidate['group']}.jpg"
        candidate['visualization'] = name
        candidate['review_status'] = 'candidate; observable-proxy tag, pending visual review'
        make_case(args.image_root / image['file_name'], candidate, gts[image['id']], dts[image['id']], names, output / name)
        manifest.append(candidate)
    write_csv(output / 'case_manifest.csv', manifest)
    save_json(output / 'case_manifest.json', manifest)
    save_json(output / 'case_summary.json', dict(cases=len(manifest), groups={g:sum(c['group'] == g for c in manifest) for g in GROUPS},
                                               operational_error_summary=errors,
                                               feature_definitions=dict(cnr='abs(mean bbox - mean ring) / max(std ring,1); ring excludes all GT',
                                                                        background_cv='std ring / max(mean ring,1)',
                                                                        texture='mean absolute horizontal neighbor difference in ring',
                                                                        bright_fraction='bbox pixel fraction above ring mean + 2*std',
                                                                        close_neighbors='center distance < 2*sqrt(this GT area)')))
    for group in GROUPS:
        members = [c for c in manifest if c['group'] == group]
        if not members:
            continue
        sheet = Image.new('RGB', (1280, len(members) * 395), '#10151d')
        for i, member in enumerate(members):
            image = Image.open(output / member['visualization']).resize((640, 395))
            sheet.paste(image, (0, i * 395))
            # A second thumbnail without annotation is deliberately omitted:
            # every panel retains original context and enlarged crop.
        sheet = sheet.crop((0, 0, 640, len(members) * 395))
        sheet.save(output / f'contact_{group}.jpg', quality=85)
    page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>SAR 困难样本案例库</title>',
            '<style>body{max-width:1360px;margin:28px auto;background:#111820;color:#e4edf7;font:16px system-ui}img{width:100%;height:auto}article{margin:24px 0 50px}a{color:#79c5ff}p{line-height:1.7}code{color:#a9d9ff}</style>',
            '<h1>SAR 困难样本案例库</h1><p>val / 最佳 EMA / 640 输入。显示原图上下文和局部放大；局部放大仅用于展示。',
            '绿色为 GT，青色为 TP，红色为背景误检，橙色定位，黄色重复，粉色分类，紫色分类与定位同时错误。score≥0.3 / IoU≥0.5。</p>',
            '<p>现象标签由可测图像代理筛选，不能等同于 SAR 物理成因。框内亮度也包含背景；低对比度/外观稀疏属于候选标签。人工抽查意见另见 visual_review.md。</p>',
            '<nav>' + ' | '.join(f'<a href="#{group}">{label}</a>' for group, label in GROUPS.items()) + '</nav>']
    for group, label in GROUPS.items():
        members = [c for c in manifest if c['group'] == group]
        page.append(f'<h2 id="{group}">{label}（{len(members)}）</h2>')
        for member in members:
            page.append(f'<article><h3>{member["case_id"]} · {html.escape(member["file_name"])} · {member["status"]}</h3>'
                        f'<img loading="lazy" src="{member["visualization"]}"><p>image_id={member["image_id"]}；'
                        f'gt_id={member.get("gt_id")}；状态：候选。详细数值见 case_manifest.csv。</p></article>')
    page.append('</html>')
    (output / 'index.html').write_text('\n'.join(page), encoding='utf-8')
    print(f'Wrote {len(manifest)} cases to {output}', flush=True)


if __name__ == '__main__':
    main()
