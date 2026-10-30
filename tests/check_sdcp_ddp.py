"""Two CPU ranks: verify SDCP margin gradients equal the global positive mean."""

import copy
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from engine.deim.deim_criterion import DEIMCriterion


def worker(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    torch.set_num_threads(1)
    torch.manual_seed(269)
    model = DistributedDataParallel(torch.nn.Linear(4, 3))
    loss_fn = DEIMCriterion(None, {}, [], num_classes=3, use_sdcp=True, sdcp_margin=.9)
    inputs = torch.randn(2, 2, 4)
    labels = torch.tensor([0, 1])
    for counts in [(1, 0), (2, 1), (0, 0)]:
        model.zero_grad(set_to_none=True)
        logits = model(inputs[rank:rank + 1])
        count = counts[rank]
        targets = [dict(labels=labels[:count])]
        indices = [(torch.arange(count), torch.arange(count))]
        value = loss_fn.loss_sdcp_margin({'sdcp_logits': logits}, targets, indices)
        value.backward()
        reference = copy.deepcopy(model.module)
        reference.zero_grad(set_to_none=True)
        all_logits = reference(inputs)
        selected = torch.cat([all_logits[index, :n] for index, n in enumerate(counts)])
        selected_labels = torch.cat([labels[:n] for n in counts])
        probabilities = selected.sigmoid()
        positive = probabilities.gather(1, selected_labels[:, None]).squeeze(1)
        others = probabilities.masked_fill(torch.nn.functional.one_hot(selected_labels, 3).bool(), -1.).amax(-1)
        expected = torch.relu(.9 - (positive - others)).square().sum() / max(sum(counts), 1)
        expected.backward()
        for name, parameter in model.module.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                       rtol=1e-5, atol=1e-7)
        if rank == 0:
            print(f'PASS positive_counts={counts}: DDP gradients equal global positive mean', flush=True)
    dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory(prefix='deim_sdcp_ddp_') as directory:
        rendezvous = (Path(directory) / 'rendezvous').as_uri()
        mp.spawn(worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == '__main__':
    main()
