"""Extract SAR-AIRcraft-1.0 and derive COCO files without shifting XML boxes."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image


def extract_archive(archive, dataset_root):
    """Strip the release's origin_data/ wrapper into a new dataset directory."""
    if dataset_root.exists():
        raise FileExistsError(f'Extraction target already exists: {dataset_root}')
    with zipfile.ZipFile(archive) as source:
        entries = []
        for info in source.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or '..' in member.parts or member.parts[0] != 'origin_data':
                raise ValueError(f'Unexpected archive path: {info.filename}')
            if len(member.parts) > 1:
                entries.append((info, Path(*member.parts[1:])))
        dataset_root.mkdir(parents=True)
        count = 0
        for info, relative in entries:
            target = dataset_root / relative
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(info) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            count += 1
            if count % 500 == 0:
                print(f'extracted {count} files', flush=True)
        print(f'extracted {count} files to {dataset_root}', flush=True)


def convert_box(values, width, height):
    """Keep source endpoints exactly: no origin shift, +1, or clipping."""
    x1, y1, x2, y2 = map(float, values)
    if not all(math.isfinite(x) for x in [x1, y1, x2, y2]):
        raise ValueError('Non-finite source box')
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f'Invalid source box {values} in {width}x{height}')
    return [x1, y1, x2 - x1, y2 - y1]


def read_splits(root):
    splits = {}
    for name in ['train', 'val', 'trainval', 'test']:
        ids = (root / 'ImageSets/Main' / f'{name}.txt').read_text(encoding='utf-8-sig').split()
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate image IDs in {name}')
        splits[name] = ids
    train, val, trainval, test = (set(splits[k]) for k in ['train', 'val', 'trainval', 'test'])
    if train & val or trainval & test or train | val != trainval:
        raise ValueError('Original split sets are inconsistent')
    return splits


def prepare_dataset(root, output_dir):
    splits = read_splits(root)
    xmls = sorted((root / 'Annotations').glob('*.xml'))
    roots = {p.stem: ET.parse(p).getroot() for p in xmls}
    if set(roots) != set(splits['trainval']) | set(splits['test']):
        raise ValueError('XML files and original split IDs differ')
    if {p.stem for p in (root / 'JPEGImages').glob('*.jpg')} != set(roots):
        raise ValueError('JPEG and XML image IDs differ')
    names = sorted({obj.findtext('name').strip() for e in roots.values() for obj in e.findall('object')})
    categories = [dict(id=i, name=name, supercategory='aircraft') for i, name in enumerate(names)]
    labels = {c['name']: c['id'] for c in categories}
    images, annotations = {}, defaultdict(list)
    dimension_counts, difficult_counts = Counter(), Counter()
    annotation_id, max_roundtrip_error = 1, 0.0
    for index, (stem, element) in enumerate(roots.items(), 1):
        image_id = int(stem)
        width, height = int(element.findtext('size/width')), int(element.findtext('size/height'))
        relative = f'JPEGImages/{stem}.jpg'
        with Image.open(root / relative) as image:
            if image.size != (width, height):
                raise ValueError(f'XML/image dimensions differ for {stem}: {(width,height)} vs {image.size}')
        images[stem] = dict(id=image_id, file_name=relative, width=width, height=height)
        dimension_counts[f'{width}x{height}'] += 1
        for object_index, obj in enumerate(element.findall('object')):
            source_box = [float(obj.findtext(f'bndbox/{key}')) for key in ['xmin', 'ymin', 'xmax', 'ymax']]
            bbox = convert_box(source_box, width, height)
            rebuilt = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            max_roundtrip_error = max(max_roundtrip_error, max(abs(a - b) for a, b in zip(source_box, rebuilt)))
            difficult = int(obj.findtext('difficult', '0'))
            difficult_counts[difficult] += 1
            annotations[stem].append(dict(id=annotation_id, image_id=image_id,
                category_id=labels[obj.findtext('name').strip()], bbox=bbox,
                area=bbox[2] * bbox[3], iscrowd=0,
                source_voc_bbox=source_box, source_object_index=object_index,
                source_difficult=difficult, source_truncated=int(obj.findtext('truncated', '0'))))
            annotation_id += 1
        if index % 500 == 0:
            print(f'converted {index}/{len(roots)} images', flush=True)
    if len({im['id'] for im in images.values()}) != len(images):
        raise ValueError('Numeric image IDs are not unique')
    data, summaries = {}, {}
    for split, ids in splits.items():
        split_images = [images[stem] for stem in ids]
        objects = [ann for stem in ids for ann in annotations[stem]]
        data[split] = dict(images=split_images, annotations=objects, categories=categories,
            info=dict(description='SAR-AIRcraft-1.0 original split, endpoint-preserving COCO conversion'))
        counts = Counter(ann['category_id'] for ann in objects)
        summaries[split] = dict(images=len(ids), objects=len(objects),
            empty_images=sum(not annotations[stem] for stem in ids),
            objects_per_class={c['name']: counts[c['id']] for c in categories})
    summary = dict(dataset='SAR-AIRcraft-1.0', dataset_root=str(root),
        unique_images=len(images), unique_objects=annotation_id - 1, categories=categories,
        dimensions=dict(dimension_counts), difficult_counts=dict(difficult_counts),
        splits=summaries, max_box_roundtrip_error=max_roundtrip_error,
        conversion='[xmin,ymin,xmax-xmin,ymax-ymin]; no +/-1, clipping, image resizing, or split resampling',
        source_files_preserved=True, train_val_test_disjoint=True, trainval_equals_train_union_val=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, content in data.items():
        (output_dir / f'{split}.json').write_text(json.dumps(content, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'stats.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'category_map.json').write_text(json.dumps(categories, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for split, values in summaries.items():
        print(f'{split}: {values["images"]} images, {values["objects"]} objects, {values["empty_images"]} empty')
    print('categories:', labels)
    print('maximum source-box reconstruction error:', max_roundtrip_error)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Extract a fresh origin_data/ release first')
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, help='Default: dataset-root/COCO; must be new')
    args = parser.parse_args()
    if args.archive:
        extract_archive(args.archive, args.dataset_root)
    prepare_dataset(args.dataset_root, args.output_dir or args.dataset_root / 'COCO')


if __name__ == '__main__':
    main()
