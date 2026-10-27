# Strip R-CNN attribution

The strip branch in `engine/deim/shallow_detail.py` adapts the `StripBlock` and
`Attention` operators from [Strip R-CNN](https://github.com/YXB-NKU/Strip-R-CNN),
by Xinbin Yuan, Zhaohui Zheng, Yuxuan Li, Xialei Liu, Li Liu, Xiang Li,
Qibin Hou and Ming-Ming Cheng. Paper: *Strip R-CNN: Large Strip Convolution for
Remote Sensing Object Detection*, AAAI 2026; [arXiv:2501.03775](https://arxiv.org/abs/2501.03775).

Source: [mmrotate/models/backbones/stripnet.py](https://github.com/YXB-NKU/Strip-R-CNN/blob/main/mmrotate/models/backbones/stripnet.py).
Upstream license: [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/),
as specified in the author's [LICENSE](https://github.com/YXB-NKU/Strip-R-CNN/blob/main/LICENSE).
That license and attribution apply to the adapted strip operator; this notice
does not replace the existing DEIM license for unrelated code. The authors
do not endorse this adaptation.

Changes: removed the MMRotate/timm wrapper dependencies; retained one residual
attention operator, without the surrounding StripNet FFN, LayerScale or backbone;
attached it to an optional stride-4 detail branch in DEIM; added a local-kernel
control with the same projections and gating. Standard PyTorch convolution
initialization is used, followed by zero affine initialization in the injection
branch's final BN. No upstream pretrained StripNet weights are loaded.
