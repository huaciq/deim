"""Build an annotation-only RegionDense bank. Run once before training; CPU only."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.data.region_bank import build_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--context-margin', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=20261110)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Use a fresh bank path: {output}')
    start = time.monotonic()
    with open(args.annotations, encoding='utf-8') as handle:
        annotation = json.load(handle)
    arrays, report = build_bank(annotation, size=args.size, margin=args.context_margin,
                                seed=args.seed, progress=lambda n: print(f'Indexed {n} images', flush=True))
    report.update(annotation_path=str(Path(args.annotations).resolve()),
                  build_seconds=time.monotonic() - start)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
