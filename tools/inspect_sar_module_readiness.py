#!/usr/bin/env python3
"""Record actual factory/optimizer/COCO-loading compatibility on CPU only.

No image reads, model forward, optimizer step, CUDA allocation, or training.
The output is experiment preparation metadata, not a training result.
"""
import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from engine.core import YAMLConfig
from engine.solver._solver import BaseSolver


def inspect(config, checkpoint=None):
    torch.manual_seed(0)
    cfg = YAMLConfig(str(config), device='cpu')
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model
    load_text = io.StringIO()
    if checkpoint:
        solver = BaseSolver(cfg)
        solver.model = model
        with contextlib.redirect_stdout(load_text):
            solver.load_tuning_state(str(checkpoint))
        del solver
    optimizer = cfg.optimizer
    criterion = cfg.criterion
    modules = dict(dsg=model.encoder.dsg, slqc=model.decoder.slqc, scs=model.decoder.scs_head)
    report = dict(config=str(config), seed=0,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  module_parameters={name:sum(p.numel() for p in module.parameters()) if module is not None else 0
                                     for name, module in modules.items()},
                  residual_initial_values={name:float(module.alpha.detach()) for name, module in modules.items()
                                           if module is not None and hasattr(module, 'alpha')},
                  encoder_class_head=list(model.decoder.enc_score_head.weight.shape),
                  decoder_class_heads=[list(head.weight.shape) for head in model.decoder.dec_score_head],
                  denoising_embedding=list(model.decoder.denoising_class_embed.weight.shape),
                  optimizer_groups=[dict(lr=group['lr'], weight_decay=group['weight_decay'],
                                         tensors=len(group['params']), parameters=sum(p.numel() for p in group['params']))
                                    for group in optimizer.param_groups],
                  scs_criterion_enabled=criterion.scs_loss is not None,
                  checkpoint=str(checkpoint) if checkpoint else None,
                  loading_log=load_text.getvalue().strip(),
                  model_forward=False, optimizer_step=False, cuda_used=False)
    del optimizer, criterion, modules, model, cfg
    gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs='+', type=Path,
                        default=[Path('configs/deim_dfine/sar_aircraft_modules') / (name + '.yml')
                                 for name in ['dsg', 'slqc', 'scs_slqc_dsg']])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for config in args.configs:
        report = inspect(config, args.checkpoint)
        results.append(report)
        (args.output / (config.stem + '.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(dict(config=str(config), parameters=report['trainable_parameters'],
                              modules=report['module_parameters'], loaded=bool(args.checkpoint))), flush=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', 'status', '--short'], text=True).splitlines()
    (args.output / 'summary.json').write_text(json.dumps(dict(git_revision=revision,
                                                             git_worktree_status=status,
                                                             torch_version=torch.__version__,
                                                             reports=results), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
