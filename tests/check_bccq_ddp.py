"""Two-rank complete-decoder BCCQ gradients vs global-batch reference, twice."""
import copy
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from test_bccq import fixture, make_model, make_criterion


def worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    torch.set_num_threads(1)
    torch.manual_seed(937)
    model = DistributedDataParallel(make_model(num_denoising=0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    features, targets = fixture()
    targets[1] = dict(labels=torch.empty(0, dtype=torch.long), boxes=torch.empty(0, 4))
    for step in range(2):
        reference = copy.deepcopy(model.module)
        optimizer.zero_grad(set_to_none=True)
        actual = model([item[rank:rank + 1] for item in features], [targets[rank]])
        sum(make_criterion()(actual, [targets[rank]]).values()).backward()
        # Both workers compute the same reference, so existing criterion
        # collectives normalize this two-image batch by its two GT objects.
        expected = reference(features, targets)
        sum(make_criterion()(expected, targets).values()).backward()
        reference_params = dict(reference.named_parameters())
        checked = 0
        for name, parameter in model.module.named_parameters():
            if name.startswith('decoder.bccq.'):
                assert parameter.grad is not None, name
                torch.testing.assert_close(parameter.grad, reference_params[name].grad,
                                           rtol=3e-3, atol=3e-5, msg=name)
                checked += 1
        optimizer.step()
        if rank == 0:
            print(f'PASS step={step}: {checked} BCCQ parameter gradients equal global-batch reference; '
                  'second rank has no GT, checkpoint enabled, find_unused_parameters=False', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='deim_bccq_ddp_') as directory:
        mp.spawn(worker, args=((Path(directory) / 'rendezvous').as_uri(),), nprocs=2, join=True)
