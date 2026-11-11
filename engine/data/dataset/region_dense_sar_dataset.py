"""COCO training dataset backed by fixed background-compatible RegionDense plans."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler

from ...core import register
from ..region_bank import tile_sizes
from ..region_composition import apply_region_plan
from .coco_dataset import CocoDetection


class RegionDensePlanSampler(DistributedSampler):
    """Ordinary baseline sampler outside active epochs, ordered plan slots inside."""

    def __iter__(self):
        self.epoch = self.dataset.epoch
        if self.dataset.plan_active:
            return iter(range(self.rank, self.dataset.output_canvases, self.num_replicas))
        return super().__iter__()

    def __len__(self):
        if self.dataset.plan_active:
            return self.dataset.output_canvases // self.num_replicas
        return super().__len__()


@register()
class RegionDenseSARCocoDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, plan_dir, plan_settings,
                 return_masks=False, remap_mscoco_category=False, keep_overflow=False):
        if return_masks or remap_mscoco_category or keep_overflow:
            raise ValueError('RegionDenseSAR requires zero-based ordinary bbox targets')
        super().__init__(img_folder, ann_file, transforms, return_masks,
                         remap_mscoco_category, keep_overflow)
        self.plan_dir = Path(plan_dir)
        self.plan_settings = plan_settings
        self.plan_metadata = json.loads((self.plan_dir / 'plan_metadata.json').read_text(encoding='utf-8'))
        if self.plan_metadata['settings'] != plan_settings:
            raise ValueError('RegionDenseSAR plan settings differ from this experiment config')
        with np.load(self.plan_dir / 'bank.npz', allow_pickle=False) as archive:
            self.source_ids = archive['image_ids'].copy()
            self.source_counts = archive['counts'].copy()
            self.source_crops = archive['crops'].copy()
        if self.ids != self.source_ids.tolist():
            raise ValueError('RegionDenseSAR plans must belong to this training split')
        self.dataset_indices = {int(i): k for k, i in enumerate(self.ids)}
        self.output_canvases = self.plan_metadata['output_canvases_per_epoch']
        self.plan_report = json.loads((self.plan_dir / 'plan_report.json').read_text(encoding='utf-8'))
        self._plan_rows = None
        self._loaded_plan_epoch = None

    @property
    def plan_active(self):
        start, stop = self.plan_settings['epochs']
        return start <= self.epoch < stop

    def make_sampler(self, batch_size, num_replicas, rank, shuffle, drop_last):
        sampler = RegionDensePlanSampler(self, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        expected = (sampler.num_samples // batch_size) * batch_size * num_replicas
        if not drop_last or batch_size * num_replicas != self.plan_metadata['global_batch_size']:
            raise ValueError('RegionDenseSAR loader must use the planned global batch and drop_last=True')
        if expected != self.output_canvases:
            raise ValueError('RegionDenseSAR plan must preserve the baseline optimizer-step budget')
        return sampler

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self._plan_rows = None
        self._loaded_plan_epoch = None
        if self.plan_active and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            summary = self.plan_report['epochs'][str(epoch)]
            print(f'RegionDenseSAR epoch={epoch}: {json.dumps(summary)}', flush=True)

    def plan_row(self, index):
        if self._loaded_plan_epoch != self.epoch:
            self._plan_rows = np.load(self.plan_dir / f'epoch_{self.epoch:03d}.npy', mmap_mode='r')
            self._loaded_plan_epoch = self.epoch
        return self._plan_rows[index]

    def load_planned_item(self, index, trace=False):
        row = self.plan_row(index)
        layout, first = int(row[0]), int(row[1])
        image, target = self.load_item(first)
        if layout < 0:
            return image, target, [], []
        size = self.plan_settings['size']
        width, height = tile_sizes(size)[layout]
        plan = []
        for slot, source in enumerate(row[1:]):
            if source < 0:
                continue
            source = int(source)
            x, y = (slot % (size // width)) * width, (slot // (size // width)) * height
            plan.append(dict(image_id=int(self.source_ids[source]), primary=slot == 0,
                             crop=tuple(map(int, self.source_crops[source, layout])),
                             offset=(x, y), counts=self.source_counts[source]))
        image, target, details = apply_region_plan(
            image, target, self, plan, size=size, context_margin=self.plan_settings['context_margin'],
            classes=self.source_counts.shape[1], dataset_indices=self.dataset_indices,
            trace=trace, keep_all_sources=True)
        return image, target, plan, details

    def __getitem__(self, index):
        if not self.plan_active:
            return super().__getitem__(index)
        image, target, _, _ = self.load_planned_item(index)
        if self._transforms is not None:
            image, target, _ = self._transforms(image, target, self)
        return image, target
