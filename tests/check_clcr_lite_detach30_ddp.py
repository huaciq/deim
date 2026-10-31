"""Two-rank DDP/AdamW transition: unchanged warmup head, then shared-head learning."""

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
from engine.solver.det_engine import _discard_inactive_clcr_lite_gradients, _set_clcr_lite_epoch


class SharedDelayedCalibration(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.decoder = nn.Module()
        self.decoder.decoder.lqe_layers = nn.ModuleList([LQE(4, 16, 2, 8) for _ in range(4)])
        self.decoder.decoder.clcr_lite = CLCRLite(detach_inputs=True, start_epoch=30)

    def forward(self, scores, corners, boxes):
        decoder = self.decoder.decoder
        return torch.stack([layer(scores[:, index], corners[:, index], boxes[:, index],
                                  clcr_lite=decoder.clcr_lite)
                            for index, layer in enumerate(decoder.lqe_layers)], dim=1)


def worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    torch.set_num_threads(1)
    torch.manual_seed(541)
    model = DistributedDataParallel(SharedDelayedCalibration(), find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.2)
    lite = model.module.decoder.decoder.clcr_lite
    initial = {name: p.detach().clone() for name, p in lite.named_parameters()}
    scores = torch.randn(2, 4, 3, 7)
    corners = torch.randn(2, 4, 3, 36)
    boxes = torch.rand(2, 4, 3, 4) * .5 + .2
    for epoch in (28, 29, 30, 31):
        _set_clcr_lite_epoch(model, epoch)
        reference = copy.deepcopy(model.module)
        optimizer.zero_grad(set_to_none=True)
        model(scores[rank:rank + 1], corners[rank:rank + 1], boxes[rank:rank + 1]).square().mean().backward()
        reference(scores, corners, boxes).square().mean().backward()
        expected = dict(reference.named_parameters())
        for name, parameter in model.module.named_parameters():
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, expected[name].grad, rtol=1e-4, atol=1e-7)
        if epoch < 30:
            assert all(p.grad.abs().sum() == 0 for p in lite.parameters())
        else:
            assert lite.calibration[-1].weight.grad.abs().sum() > 0
            if epoch > 30:
                assert lite.calibration[0].weight.grad.abs().sum() > 0
                assert lite.beta.grad.abs() > 0
        _discard_inactive_clcr_lite_gradients(model)
        optimizer.step()
        if epoch < 30:
            for name, parameter in lite.named_parameters():
                torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
                assert parameter not in optimizer.state
        if rank == 0:
            print(f'PASS epoch={epoch}: active={lite.active}, four shared layers; '
                  'DDP gradients match serial global mean; '
                  + ('inactive freeze verified' if epoch < 30 else 'active learning verified'), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    temp_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix='deim_detach30_ddp_', dir=str(temp_root)) as directory:
        run = Path(directory).resolve()
        if run.parent != temp_root or not run.name.startswith('deim_detach30_ddp_'):
            raise RuntimeError('Unexpected temporary directory')
        mp.spawn(worker, args=((run / 'rendezvous').as_uri(),), nprocs=2, join=True)
