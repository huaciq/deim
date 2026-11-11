"""Copy intact regions at ordinary resize scale, with aligned bbox annotations."""

import numpy as np
import torch
from PIL import Image

from ._misc import convert_to_tv_tensor
from .region_bank import context_boxes, visible_and_preserved


def apply_region_plan(image, target, dataset, plan, *, size, context_margin,
                      classes, dataset_indices, trace=False, keep_all_sources=False):
    canvas = Image.new(image.mode, (size, size))
    output_boxes, output_labels, output_crowd, details = [], [], [], []
    for sample in plan:
        if sample['primary']:
            source, annotation = image, target
        else:
            source, annotation = dataset.load_item(dataset_indices[sample['image_id']])
        if 'masks' in annotation or 'keypoints' in annotation or annotation['labels'].ndim != 1:
            raise ValueError('RegionDense currently supports ordinary bbox targets only')
        width, height = source.size
        source = source.resize((size, size), Image.Resampling.BILINEAR)
        boxes = annotation['boxes'].as_subclass(torch.Tensor).double().numpy()
        boxes = boxes * np.array([size / width, size / height] * 2)
        contexts = context_boxes(boxes, size, context_margin)
        visible, preserved = visible_and_preserved(boxes, contexts, sample['crop'], tolerance=1e-3)
        if np.any(visible & ~preserved) or ((sample['primary'] or keep_all_sources) and not visible.all()):
            raise ValueError('RegionDense crop would cut a target or its protected context')
        labels = annotation['labels'][torch.from_numpy(visible)]
        histogram = np.bincount(labels.numpy(), minlength=classes)
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
    result = dict(boxes=convert_to_tv_tensor(boxes, 'boxes', spatial_size=(size, size)),
                  labels=torch.cat(output_labels), iscrowd=torch.cat(output_crowd),
                  area=(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                  image_id=target['image_id'].clone(), idx=target['idx'].clone(),
                  orig_size=torch.tensor([size, size]))
    return canvas, result, details
