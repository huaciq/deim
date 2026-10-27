"""CPU two-rank verification of MPQ's globally class-balanced DDP gradients.

Run: torchrun --standalone --nproc_per_node=2 tests/check_mpq_ddp.py
Checks an empty rank, unequal class frequencies, and pre-activation gradients.
No detector, dataset, GPU allocation, or experiment checkpoint is used.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from engine.deim.query_prototypes import QueryPrototypeHead, MatchedQueryPrototypeLoss


def case(head, queries, labels, empty):
    out = head(queries)
    out['pred_boxes'] = torch.tensor([[[.5, .5, .2, .2]] * queries.shape[1]])
    labels = labels[:0] if empty else labels
    targets = [dict(labels=labels, boxes=torch.tensor([[.5, .5, .2, .2]]).repeat(len(labels), 1))]
    ids = torch.arange(len(labels))
    return out, targets, [(ids, ids)]


def main():
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    torch.set_num_threads(1)
    for epoch, empty_second in [(0, False), (54, False), (54, True)]:
        torch.manual_seed(12)
        head = QueryPrototypeHead(8, 3, 4, [1, 1, 4])
        model = DistributedDataParallel(head)
        torch.manual_seed(13)
        queries = torch.randn(2, 3, 8)
        labels = [torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2])]
        out, targets, indices = case(model, queries[rank:rank + 1], labels[rank], rank == 1 and empty_second)
        criterion = MatchedQueryPrototypeLoss(3)
        value = criterion(out, targets, indices, epoch=epoch)
        value.backward()
        parallel_grads = {name: p.grad.clone() for name, p in head.named_parameters()}
        # Explicit serial reference uses class means across the complete batch.
        reference_head = QueryPrototypeHead(8, 3, 4, [1, 1, 4])
        reference_head.load_state_dict(head.state_dict())
        full_logits = reference_head(queries)['pred_mpq_logits']
        sample_logits = torch.cat([full_logits[0], full_logits[1, :0] if empty_second else full_logits[1]])
        sample_labels = torch.cat([labels[0], labels[1][:0] if empty_second else labels[1]])
        ce = torch.nn.functional.cross_entropy(sample_logits, sample_labels, reduction='none')
        reference = torch.stack([ce[sample_labels == c].mean() for c in sample_labels.unique()]).mean()
        if epoch == 0: reference = reference * 0
        reference.backward()
        for name, p in reference_head.named_parameters():
            torch.testing.assert_close(parallel_grads[name], p.grad, rtol=2e-5, atol=1e-6)
        if rank == 0: print(f'PASS epoch={epoch} empty_second_rank={empty_second}: global class means and gradients', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
