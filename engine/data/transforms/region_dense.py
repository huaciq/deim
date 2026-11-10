"""RegionDense: add intact regions at ordinary training scale, retaining the primary GTs."""

import random

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor
from ..region_bank import RegionBank, context_boxes, visible_and_preserved


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
        canvas = Image.new(image.mode, (self.size, self.size))
        output_boxes, output_labels, output_crowd, details = [], [], [], []
        for sample in plan:
            if sample['primary']:
                source, annotation = image, target
            else:
                source, annotation = dataset.load_item(self.dataset_indices[sample['image_id']])
            if 'masks' in annotation or 'keypoints' in annotation or annotation['labels'].ndim != 1:
                raise ValueError('RegionDense currently supports ordinary bbox targets only')
            width, height = source.size
            source = source.resize((self.size, self.size), Image.Resampling.BILINEAR)
            boxes = annotation['boxes'].as_subclass(torch.Tensor).double().numpy()
            boxes = boxes * np.array([self.size / width, self.size / height] * 2)
            contexts = context_boxes(boxes, self.size, self.context_margin)
            visible, preserved = visible_and_preserved(boxes, contexts, sample['crop'], tolerance=1e-3)
            if np.any(visible & ~preserved) or (sample['primary'] and not visible.all()):
                raise ValueError('RegionDense crop would cut a target or its protected context')
            labels = annotation['labels'][torch.from_numpy(visible)]
            histogram = np.bincount(labels.numpy(), minlength=self.bank.classes)
            if not np.array_equal(histogram, sample['counts']):
                raise ValueError('RegionDense bank annotations do not match the loaded image')
            x, y = sample['offset']
            left, top, _, _ = sample['crop']
            translated = boxes[visible] + np.array([x - left, y - top] * 2)
            canvas.paste(source.crop(sample['crop']), (x, y))
            output_boxes.append(torch.as_tensor(translated, dtype=torch.float32))
            output_labels.append(labels)
            output_crowd.append(annotation['iscrowd'][torch.from_numpy(visible)])
            if trace:
                details.append(dict(image_id=sample['image_id'], primary=sample['primary'],
                                    crop=sample['crop'], offset=sample['offset'],
                                    gt_indices=np.flatnonzero(visible).tolist(),
                                    input_boxes=boxes[visible].tolist(),
                                    output_boxes=translated.tolist(), labels=labels.tolist()))
        boxes = torch.cat(output_boxes)
        result = dict(boxes=convert_to_tv_tensor(boxes, 'boxes', spatial_size=(self.size, self.size)),
                      labels=torch.cat(output_labels), iscrowd=torch.cat(output_crowd),
                      area=(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                      image_id=target['image_id'].clone(), idx=target['idx'].clone(),
                      orig_size=torch.tensor([self.size, self.size]))
        return canvas, result, details

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
