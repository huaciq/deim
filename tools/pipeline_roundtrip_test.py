#!/usr/bin/env python3
"""
Locate the source of the constant x offset by measuring each pipeline stage.

bias_shift_search showed that shifting every detection by -0.75 source pixels
is worth +5.12 AP, with dy = 0 exactly. The offset is constant in pixels,
independent of object size, and present in every category, so it is a
coordinate-convention bug rather than a model limitation. This test isolates
which stage introduces it by pushing a synthetic box with a known position
through each stage and measuring the round-trip error.

Stages tested independently, so a defect in one cannot hide behind another:
  resize      256 -> 640 and back
  hflip       mirror and mirror again
  normalize   xyxy pixel -> normalised cxcywh -> pixel
  mosaic      placement and the final resize

Run on the server (needs torchvision and PIL, no GPU):

python tools/pipeline_roundtrip_test.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from PIL import Image

from engine.data.transforms.functional import hflip, resize as deim_resize


def make_box(x1, y1, w, h):
    return torch.tensor([[x1, y1, x1 + w, y1 + h]], dtype=torch.float32)


def centre(b):
    return ((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2)


def report(name, before, after, tol=1e-4):
    cx0, cy0 = centre(before)
    cx1, cy1 = centre(after)
    dx = (cx1 - cx0).item()
    dy = (cy1 - cy0).item()
    flag = 'OK' if max(abs(dx), abs(dy)) <= tol else '**OFFSET**'
    print(f"{name:34s} dx={dx:+7.4f}  dy={dy:+7.4f}  {flag}")


def main():
    print("stage-isolated round-trip tests (all offsets in source pixels)\n")
    box = make_box(100.0, 90.0, 12.0, 10.0)

    # --- 1. DEIM resize 256 -> 640, then map back ---
    img = Image.new('RGB', (256, 256))
    target = {'boxes': box.clone()}
    out_img, out_target = deim_resize(img, target, [640, 640])
    b640 = out_target['boxes'].clone()
    back = b640 * torch.tensor([256 / 640, 256 / 640, 256 / 640, 256 / 640])
    report("deim resize 256->640->256", box, back)

    # --- 2. torchvision Resize (used by Mosaic) ---
    tv_resize = T.Resize(size=320)
    tv_target = {'boxes': box.clone()}
    out_img, out_target = tv_resize(img, tv_target)
    b320 = out_target['boxes'].clone()
    back = b320 * torch.tensor([256 / 320, 256 / 320, 256 / 320, 256 / 320])
    report("tv Resize 256->320->256", box, back)

    # --- 3. horizontal flip twice ---
    flipped, ft = hflip(img, {'boxes': box.clone()})
    twice, tt = hflip(flipped, {'boxes': ft['boxes'].clone()})
    report("deim hflip applied twice", box, tt['boxes'])

    # --- 4. flip once, then check the mirror identity ---
    once, ot = hflip(img, {'boxes': box.clone()})
    cx0, _ = centre(box)
    cx1, _ = centre(ot['boxes'])
    expected = 256.0 - cx0
    print(f"{'deim hflip single, mirror about w':34s} got {cx1.item():.4f}  "
          f"expected {expected.item():.4f}  "
          f"{'OK' if abs((cx1 - expected).item()) < 1e-4 else '**OFFSET**'}")
    expected_m1 = 255.0 - cx0
    print(f"{'  (mirror about w-1 convention)':34s} got {cx1.item():.4f}  "
          f"expected {expected_m1.item():.4f}  "
          f"{'matches w-1' if abs((cx1 - expected_m1).item()) < 1e-4 else 'does not match w-1'}")

    # --- 5. normalization round trip (pixel xyxy -> norm cxcywh -> pixel) ---
    H = W = 640
    b = make_box(250.0, 225.0, 30.0, 25.0)
    norm = b / torch.tensor([W, H, W, H])
    cxcywh = torch.empty_like(norm)
    cxcywh[:, 0] = (norm[:, 0] + norm[:, 2]) / 2
    cxcywh[:, 1] = (norm[:, 1] + norm[:, 3]) / 2
    cxcywh[:, 2] = norm[:, 2] - norm[:, 0]
    cxcywh[:, 3] = norm[:, 3] - norm[:, 1]
    x1 = (cxcywh[:, 0] - cxcywh[:, 2] / 2) * W
    y1 = (cxcywh[:, 1] - cxcywh[:, 3] / 2) * H
    x2 = (cxcywh[:, 0] + cxcywh[:, 2] / 2) * W
    y2 = (cxcywh[:, 1] + cxcywh[:, 3] / 2) * H
    back = torch.stack([x1, y1, x2, y2], 1)
    report("normalize cxcywh round trip", b, back)

    # --- 6. integer/half-pixel resize target ---
    for size in (640, 512, 320):
        img2 = Image.new('RGB', (255, 255))
        t2 = {'boxes': make_box(100.0, 90.0, 12.0, 10.0)}
        _, ot2 = deim_resize(img2, t2, [size, size])
        r = size / 255.0
        back2 = ot2['boxes'] * torch.tensor([1 / r] * 4)
        report(f"deim resize 255->{size}->255", t2['boxes'], back2)

    print("\ninterpretation:")
    print("  any **OFFSET** above localises the bug to that stage.")
    print("  if every stage is OK, the offset is introduced inside the model")
    print("  (reference-point or grid construction), not in the data pipeline.")


if __name__ == '__main__':
    main()
