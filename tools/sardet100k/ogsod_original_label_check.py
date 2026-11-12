"""Mirror-copy label gaps in the untouched OGSOD-1.0 release, labels read from the zip itself.

Checks the SAR and optical (rgb) train labels, and whether the two label sets agree.
Pairs are found on the SAR train images (their 2023 timestamps match the zip).
Usage: ogsod_original_label_check.py OGSOD_ZIP EXTRACTED_SAR_ROOT OUT_JSON
"""
import json
import multiprocessing as mp
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ogsod_mirror_offset_check import W, find_pairs, label_gaps  # noqa: E402

ZIP, SAR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])


def read_labels(archive, modality):
    out = {}
    prefix = f'OGSOD-1.0/{modality}/labels/train/'
    for name in archive.namelist():
        if not name.startswith(prefix) or not name.endswith('.txt'):
            continue
        boxes = defaultdict(list)
        for line in archive.read(name).decode().splitlines():
            p = line.split()
            if len(p) >= 5:
                c, cx, cy, w, h = int(p[0]), *(float(v) * W for v in p[1:5])
                boxes[c].append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        out[Path(name).stem] = boxes
    return out


def main():
    archive = zipfile.ZipFile(ZIP)
    sar, rgb = read_labels(archive, 'sar'), read_labels(archive, 'rgb')
    report = dict(label_files=dict(sar=len(sar), rgb=len(rgb)))
    same = differ = 0
    for stem, boxes in sar.items():
        other = rgb.get(stem)
        if other is None:
            continue
        a = sorted(map(tuple, (b for v in boxes.values() for b in v)))
        b = sorted(map(tuple, (b for v in other.values() for b in v)))
        if len(a) == len(b) and np.allclose(a, b, atol=1e-3):
            same += 1
        else:
            differ += 1
    report['sar_vs_rgb_label_files'] = dict(identical=same, different=differ)
    print(report, flush=True)
    paths = sorted(str(p) for p in (SAR / 'images/train').iterdir())
    with mp.Pool(12) as pool:
        found = find_pairs(pool, paths)
    for modality, labels in (('sar', sar), ('rgb', rgb)):
        boxes_of = {p: labels.get(Path(p).stem) for p in paths if Path(p).stem in labels}
        report[f'{modality}_train_label_gaps'] = label_gaps(found, boxes_of, paths)
    OUT.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
