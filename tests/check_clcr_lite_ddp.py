"""Verify four LQE layers sharing Lite produce the expected two-rank DDP gradient."""

import copy
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from engine.deim.dfine_decoder import CLCRLite, LQE


class SharedCalibration(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([LQE(4, 16, 2, 8) for _ in range(4)])
        self.clcr_lite = CLCRLite()

    def forward(self, scores, corners, boxes):
        return torch.stack([layer(scores[:, index], corners[:, index], boxes[:, index], clcr_lite=self.clcr_lite)
                            for index, layer in enumerate(self.layers)], dim=1)


def worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    torch.set_num_threads(1)
    torch.manual_seed(311)
    model = DistributedDataParallel(SharedCalibration())
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    scores = torch.randn(2, 4, 3, 7)
    corners = torch.randn(2, 4, 3, 36)
    boxes = torch.rand(2, 4, 3, 4) * .5 + .2
    for step in range(2):
        reference = copy.deepcopy(model.module)
        optimizer.zero_grad(set_to_none=True)
        actual = model(scores[rank:rank+1], corners[rank:rank+1], boxes[rank:rank+1])
        actual.square().mean().backward()
        expected = reference(scores, corners, boxes)
        expected.square().mean().backward()
        expected_params = dict(reference.named_parameters())
        for name, parameter in model.module.named_parameters():
            torch.testing.assert_close(parameter.grad, expected_params[name].grad, rtol=1e-4, atol=1e-7)
        lite = model.module.clcr_lite
        assert len([key for key in model.module.state_dict() if key.startswith('clcr_lite.')]) == 5
        assert lite.calibration[-1].weight.grad.abs().sum() > 0
        if step:
            assert lite.calibration[0].weight.grad.abs().sum() > 0
            assert lite.beta.grad.abs() > 0
        optimizer.step()
        if rank == 0:
            print(f'PASS step={step}: shared four-layer Lite DDP gradients equal serial global mean', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='deim_clcr_lite_ddp_') as directory:
        mp.spawn(worker, args=((Path(directory) / 'rendezvous').as_uri(),), nprocs=2, join=True)
