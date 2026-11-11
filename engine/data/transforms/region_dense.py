"""RegionDense: add intact regions at ordinary training scale, retaining the primary GTs."""

import random

import torchvision.transforms.v2 as T

from ...core import register
from ..region_bank import RegionBank
from ..region_composition import apply_region_plan


@register()
class RegionDense(T.Transform):
    def __init__(self, bank_path, size=640, context_margin=1.0,
                 probability=0.5, epochs=(4, 20), max_targets=300):
        super().__init__()
        self.bank_path = bank_path
        self.size = size
        self.context_margin = context_margin
        self.probability = probability
        self.epochs = epochs
        self.max_targets = max_targets
        self.bank = None
        self.dataset_indices = None

    def load_bank(self, dataset):
        if self.bank is None:
            self.bank = RegionBank(self.bank_path)
            if self.bank.size != self.size or self.bank.margin != self.context_margin:
                raise ValueError('RegionDense configuration and region bank geometry differ')
            self.dataset_indices = {int(v): k for k, v in enumerate(dataset.ids)}
            if set(self.dataset_indices) != set(self.bank.id_to_index):
                raise ValueError('RegionDense bank must be built from this training split')

    def apply_plan(self, image, target, dataset, plan, trace=False):
        return apply_region_plan(image, target, dataset, plan, size=self.size,
                                 context_margin=self.context_margin, classes=self.bank.classes,
                                 dataset_indices=self.dataset_indices, trace=trace)

    def forward(self, *inputs):
        image, target, dataset = inputs if len(inputs) > 1 else inputs[0]
        if not self.epochs[0] <= dataset.epoch < self.epochs[1] or random.random() >= self.probability:
            return image, target, dataset
        self.load_bank(dataset)
        plan = self.bank.plan(int(target['image_id'].item()), max_targets=self.max_targets)
        if plan is None:
            return image, target, dataset
        image, target, _ = self.apply_plan(image, target, dataset, plan)
        return image, target, dataset
