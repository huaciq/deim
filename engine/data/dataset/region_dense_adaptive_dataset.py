"""Read immutable scale-preserving plans with equal or unequal partitions."""

import numpy as np

from ...core import register
from ..region_bank import tile_sizes
from ..region_dense_adaptive import centered_crop
from ..region_composition import apply_region_plan
from .region_dense_sar_dataset import RegionDenseSARCocoDetection


@register()
class RegionDenseAdaptiveCocoDetection(RegionDenseSARCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, plan_dir, plan_settings,
                 return_masks=False, remap_mscoco_category=False, keep_overflow=False):
        super().__init__(img_folder, ann_file, transforms, plan_dir, plan_settings,
                         return_masks, remap_mscoco_category, keep_overflow)
        if self.plan_metadata['version'] != 3:
            raise ValueError('Adaptive partitions require version 3 plans')
        with np.load(self.plan_dir / 'bank.npz', allow_pickle=False) as archive:
            self.protected_regions = archive['regions'].copy()

    def load_planned_item(self, index, trace=False):
        row = self.plan_row(index)
        layout, first, cut = int(row[0]), int(row[1]), int(row[5])
        image, target = self.load_item(first)
        if layout < 0:
            return image, target, [], []
        size = self.plan_settings['size']
        plan = []
        for slot, source in enumerate(row[1:5]):
            if source < 0:
                continue
            source = int(source)
            if layout < 3:
                width, height = tile_sizes(size)[layout]
                crop = tuple(map(int, self.source_crops[source, layout]))
                offset = (slot % (size // width) * width, slot // (size // width) * height)
            else:
                width, height = size, size
                length = cut if slot == 0 else size - cut
                if layout == 3:
                    width = length
                    offset = (0 if slot == 0 else cut, 0)
                elif layout == 4:
                    height = length
                    offset = (0, 0 if slot == 0 else cut)
                else:
                    raise ValueError(f'Unknown adaptive layout: {layout}')
                crop = centered_crop(self.protected_regions[source], width, height, size)
            plan.append(dict(image_id=int(self.source_ids[source]), primary=slot == 0,
                             crop=crop, offset=offset, counts=self.source_counts[source]))
        image, target, details = apply_region_plan(
            image, target, self, plan, size=size, context_margin=self.plan_settings['context_margin'],
            classes=self.source_counts.shape[1], dataset_indices=self.dataset_indices,
            trace=trace, keep_all_sources=True)
        return image, target, plan, details
