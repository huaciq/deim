"""Compare checkpoint on/off with the REAL shared trunk under two-rank SyncBN.

Run on the requested GPUs:
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc_per_node=2 tests/check_orientation_fusion_v2_ddp.py

This checks the specific recomputation contract (outputs, every gradient, BN
buffers, optimizer updates), in FP32 and AMP. It is not an AP experiment.
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
    for amp in (False, True):
        expected = []
        for checkpointed in (False, True):
            trunk.load_state_dict(initial)
            trunk.fusion.checkpoint_views = checkpointed
            optimizer = torch.optim.SGD(trunk.parameters(), lr=1e-4)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                    features = ddp(images)
                loss = sum(level.float().square().mean() for level in features)
                loss.backward()
                assert first_bn.num_batches_tracked.item() == 2 * (step + 1)
                gradients = {name: p.grad.detach().clone() for name, p in trunk.named_parameters() if p.grad is not None}
                assert len(gradients) == len(list(trunk.parameters())), 'An encoded parameter has no gradient'
                optimizer.step()
                actual = dict(features=[x.detach().clone() for x in features], gradients=gradients,
                              state=copy.deepcopy(trunk.state_dict()))
                if not checkpointed:
                    expected.append(actual)
                else:
                    # CUDA convolution and SyncBN may select different kernels
                    # for the original forward and its checkpoint replay. The
                    # comparison checks the same optimization trajectory within
                    # a CUDA numerical tolerance, rather than CPU bit equality.
                    tolerances = dict(rtol=1e-2, atol=1e-4) if amp else dict(rtol=5e-3, atol=5e-5)
                    for a, b in zip(actual['features'], expected[step]['features']):
                        torch.testing.assert_close(a, b, **tolerances)
                    for kind in ('gradients', 'state'):
                        for name, value in actual[kind].items():
                            torch.testing.assert_close(value, expected[step][kind][name], msg=f'{kind}.{name}', **tolerances)
                    if rank == 0:
                        print(json.dumps(dict(result='PASS', amp=amp, step=step,
                                              compared_gradients=len(gradients), bn_updates=first_bn.num_batches_tracked.item())), flush=True)
        del expected
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
