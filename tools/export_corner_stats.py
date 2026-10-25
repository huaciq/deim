#!/usr/bin/env python3
"""Export FDR edge moments in the nonuniform projection coordinate system."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from tqdm import tqdm

from engine.core import YAMLConfig
from engine.deim.dfine_utils import distance2bbox, weighting_function


def distribution_moments(corners, project):
    """Return the mean/std of projected values, not of integer bin indices."""
    prob = corners.reshape(*corners.shape[:-1], 4, project.numel()).softmax(-1)
    mean = (prob * project).sum(-1)
    variance = (prob * (project - mean.unsqueeze(-1)).square()).sum(-1)
    return mean, variance.clamp_min(0).sqrt()


def projection_parameters(decoder):
    reg_max = int(decoder.reg_max)
    up = decoder.up.detach().float().cpu()
    # weighting_function concatenates 1-D tensors; keep the model parameter's
    # shape here and convert to a scalar only for decoding/serialization.
    scale = decoder.reg_scale.detach().float().cpu()
    project = weighting_function(reg_max, up, scale).float().cpu()
    return reg_max, up, float(scale.item()), project


def main(args):
    torch.manual_seed(args.seed)
    cfg = YAMLConfig(args.config, resume=args.resume)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = cfg.model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint['ema']['module'] if checkpoint.get('ema') is not None else checkpoint['model']
    model.load_state_dict(state)
    model.eval()

    reg_max, up, reg_scale, project = projection_parameters(model.decoder)
    fields = {key: [] for key in ['image_id', 'query_idx', 'score', 'label',
                                 'box', 'ref', 'mean', 'std']}
    reconstruction_error, n_images = 0.0, 0
    with torch.no_grad():
        for samples, targets in tqdm(cfg.val_dataloader, desc='export corners'):
            outputs = model(samples.to(device))
            corners = outputs['pred_corners'].detach().float().cpu()
            refs = outputs['ref_points'].detach().float().cpu()
            boxes = outputs['pred_boxes'].detach().float().cpu()
            logits = outputs['pred_logits'].detach().float().cpu()
            mean, std = distribution_moments(corners, project)
            reconstructed = distance2bbox(refs, mean, reg_scale)
            difference = (reconstructed - boxes).abs()
            if not torch.isfinite(difference).all():
                raise RuntimeError('Non-finite distribution/box reconstruction')
            reconstruction_error = max(reconstruction_error, difference.max().item())
            n_images += boxes.shape[0]
            scores, labels = logits.sigmoid().max(-1)
            for i, target in enumerate(targets):
                keep = (scores[i] > args.score_thr).nonzero(as_tuple=True)[0]
                if keep.numel() > args.topk:
                    keep = keep[scores[i, keep].topk(args.topk).indices]
                if keep.numel() == 0:
                    continue
                fields['image_id'].append(np.full(keep.numel(), int(target['image_id'].item()), np.int64))
                fields['query_idx'].append(keep.numpy().astype(np.int32))
                fields['score'].append(scores[i, keep].numpy())
                fields['label'].append(labels[i, keep].numpy().astype(np.int32))
                for key, tensor in [('box', boxes), ('ref', refs), ('mean', mean), ('std', std)]:
                    fields[key].append(tensor[i, keep].numpy())

    print(f'images: {n_images}')
    print(f'max |mean-decoded box - pred_boxes| = {reconstruction_error:.9e}')
    if reconstruction_error >= 1e-4:
        raise RuntimeError('Reconstruction mismatch: stop before interpreting FDR statistics')
    arrays = {}
    for key, values in fields.items():
        dtype = np.int64 if key == 'image_id' else np.int32 if key in ['query_idx', 'label'] else np.float32
        shape = (0, 4) if key in ['box', 'ref', 'mean', 'std'] else (0,)
        arrays[key] = np.concatenate(values, axis=0).astype(dtype) if values else np.empty(shape, dtype)
    np.savez_compressed(args.output, **arrays, project=project.numpy(), up=up.numpy(),
                        reg_scale=np.float64(reg_scale), reg_max=np.int64(reg_max),
                        recon_err=np.float64(reconstruction_error), n_images=np.int64(n_images),
                        score_threshold=np.float64(args.score_thr), topk=np.int64(args.topk))
    metadata = dict(config=args.config, checkpoint=args.resume, seed=args.seed,
                    device=str(device), precision='float32', images=n_images,
                    records=len(arrays['image_id']), reg_max=reg_max, reg_scale=reg_scale,
                    up=up.tolist(), project=project.tolist(), reconstruction_error=reconstruction_error,
                    score_threshold=args.score_thr, topk=args.topk,
                    units='nonuniform projected edge distance, not integer bin index')
    Path(args.output).with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f"records: {len(arrays['image_id'])}")
    print(f'project support: [{project.min().item()}, {project.max().item()}]')
    print(f'saved {args.output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--resume', required=True)
    parser.add_argument('-o', '--output', default='corners.npz')
    parser.add_argument('--score-thr', type=float, default=0.05)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())
