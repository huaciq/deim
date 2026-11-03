#!/usr/bin/env python3
"""Write a COCO-pretrained D-FINE/DEIM checkpoint whose class heads start from one COCO class.

train.py -t cannot map the 80 COCO classes onto SAR-AIRcraft's 7, so the class heads
would start random. This copies the chosen COCO row (default: airplane, contiguous
label 4) into every target class of the encoder/decoder score heads and of the
denoising class embedding, keeping its padding row. All other weights are unchanged.
Use the output with train.py -t (e.g. INITIAL_WEIGHTS for run_sar_module_experiment.sh).
"""

import argparse
from pathlib import Path

import torch

AIRPLANE = 4  # mscoco category 5 'airplane' -> contiguous label 4 with remap_mscoco_category


def adapt(state, num_classes, source_index=AIRPLANE):
    state = dict(state)
    changed = []
    for key, value in list(state.items()):
        is_score = key.startswith(('decoder.enc_score_head.', 'decoder.dec_score_head.'))
        if is_score and value.shape[0] == 80:
            state[key] = value[source_index:source_index + 1].repeat(num_classes, *[1] * (value.dim() - 1)).clone()
            changed.append(key)
        elif key == 'decoder.denoising_class_embed.weight' and value.shape[0] == 81:
            rows = value[source_index:source_index + 1].repeat(num_classes, 1)
            state[key] = torch.cat([rows, value[-1:]], 0).clone()  # last row is the padding index
            changed.append(key)
    if not changed:
        raise ValueError('no 80-class head parameters found')
    return state, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('source', type=Path, help='COCO-pretrained checkpoint (80 classes)')
    parser.add_argument('output', type=Path, help='new checkpoint path (must not exist)')
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--source-index', type=int, default=AIRPLANE)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.source, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    state, changed = adapt(state, args.num_classes, args.source_index)
    torch.save(dict(model=state, source=str(args.source.resolve()), source_index=args.source_index,
                    num_classes=args.num_classes, adapted=changed), args.output)
    print(f'wrote {args.output} ({len(changed)} head tensors from COCO label {args.source_index})')


if __name__ == '__main__':
    main()
