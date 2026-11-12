"""Compare checkpoint on/off with the REAL shared trunk under two-rank SyncBN.

Run on the requested GPUs:
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 tests/check_orientation_fusion_v2_ddp.py

This checks the specific recomputation contract (outputs, gradients, BN
buffers), in FP32 and AMP, against the run-to-run drift of CUDA training
(see the comment in main). CPU tests cover exact gradients. It is not an AP
experiment.
"""
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from engine.core import YAMLConfig


class SharedTrunk(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.projections = model.encoder.input_proj
        self.fusion = model.orientation_fusion

    def encode(self, images):
        return [project(feature) for project, feature in zip(self.projections, self.backbone(images))]

    def forward(self, images):
        return self.fusion(images, self.encode, checkpoint_modules=(self.backbone, self.projections))


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    torch.set_num_threads(2)
    torch.manual_seed(937)
    cfg = YAMLConfig('configs/deim_dfine/deim_hgnetv2_m_sardet100k_orientation_fusion_v2_36e_b32.yml')
    trunk = nn.SyncBatchNorm.convert_sync_batchnorm(SharedTrunk(cfg.model)).cuda(rank).train()
    # Exercise nonconstant gates as well as both feature branches.
    for gate in trunk.fusion.gates:
        nn.init.normal_(gate[-1].weight, std=0.02)
    ddp = DDP(trunk, device_ids=[rank], find_unused_parameters=False)
    initial = copy.deepcopy(trunk.state_dict())
    generator = torch.Generator(device=f'cuda:{rank}').manual_seed(200 + rank)
    images = torch.rand(2, 3, 128, 128, generator=generator, device=rank)
    first_bn = next(m for m in trunk.backbone.modules() if isinstance(m, nn.SyncBatchNorm))
    def trajectory(checkpointed, amp):
        trunk.load_state_dict(initial)
        trunk.fusion.checkpoint_views = checkpointed
        optimizer = torch.optim.SGD(trunk.parameters(), lr=1e-4)
        steps = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                features = ddp(images)
            loss = sum(level.float().square().mean() for level in features)
            loss.backward()
            assert first_bn.num_batches_tracked.item() == 2 * (step + 1)
            gradients = {name: p.grad.detach().clone() for name, p in trunk.named_parameters() if p.grad is not None}
            assert len(gradients) == len(list(trunk.parameters())), 'An encoded parameter has no gradient'
            for name, value in gradients.items():
                assert torch.isfinite(value).all(), f'non-finite gradient: {name}'
            optimizer.step()
            buffers = {name: value.detach().clone() for name, value in trunk.state_dict().items()
                       if any(token in name for token in ('running_mean', 'running_var', 'num_batches_tracked'))}
            steps.append(dict(features=[x.detach().float().clone() for x in features], gradients=gradients, buffers=buffers))
        return steps

    def feature_gap(a, b):
        return max(float((x - y).abs().max()) for x, y in zip(a['features'], b['features']))

    def gradient_gap(a, b):
        diff = sum(float((a['gradients'][n] - b['gradients'][n]).float().norm()) ** 2 for n in a['gradients']) ** .5
        return diff / sum(float(g.float().norm()) ** 2 for g in a['gradients'].values()) ** .5

    def buffer_gap(a, b):
        for name, value in a['buffers'].items():
            if 'num_batches_tracked' in name:
                assert torch.equal(value, b['buffers'][name]), f'BN.{name}'
        return max(float((value.float() - b['buffers'][name].float()).abs().max())
                   for name, value in a['buffers'].items() if 'num_batches_tracked' not in name)

    # CUDA training is not bit-reproducible: TF32 convolutions and non-deterministic cuDNN backward kernels make two
    # identical runs drift after the first optimizer step (measured on the RTX 4090s: step-1 features differ by
    # ~3e-2 between two runs without checkpointing, ~8e-5 with TF32 off). So the checkpointed trajectory is compared
    # with the plain one RELATIVE to a repeat of the plain one; a real replay error (wrong BN statistics, a missing
    # gradient path) is orders of magnitude larger than that drift.
    for amp in (False, True):
        plain, repeat, checkpointed = trajectory(False, amp), trajectory(False, amp), trajectory(True, amp)
        floor = dict(feature=1e-3, gradient=1e-3, buffer=1e-3) if amp else dict(feature=1e-5, gradient=1e-4, buffer=1e-5)
        for step in range(2):
            gaps = dict(
                feature=(feature_gap(checkpointed[step], plain[step]), feature_gap(repeat[step], plain[step])),
                gradient=(gradient_gap(checkpointed[step], plain[step]), gradient_gap(repeat[step], plain[step])),
                buffer=(buffer_gap(checkpointed[step], plain[step]), buffer_gap(repeat[step], plain[step])))
            for kind, (replayed, control) in gaps.items():
                assert replayed <= 3 * control + floor[kind], f'amp={amp} step={step} {kind}: checkpointed {replayed:.2e} vs repeat {control:.2e}'
            if rank == 0:
                print(json.dumps(dict(result='PASS', amp=amp, step=step, compared_gradients=len(plain[step]['gradients']),
                                      bn_updates=2 * (step + 1),
                                      checkpointed_vs_plain={k: f'{v[0]:.1e}' for k, v in gaps.items()},
                                      repeat_vs_plain={k: f'{v[1]:.1e}' for k, v in gaps.items()})), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
