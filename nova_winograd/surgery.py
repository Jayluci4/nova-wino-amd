"""
Model surgery: replace nn.Conv2d layers with NOVA Winograd equivalents.
"""

import torch.nn as nn
from nova_winograd.conv2d import NovaWinogradConv2d, NovaWinogradConv2dTrainable


def replace_conv2d_with_nova(model, trainable=False, verbose=True):
    """Replace all eligible nn.Conv2d layers with NovaWinogradConv2d.

    Eligible layers: kernel_size=3, stride=1, dilation=1, groups=1.
    Stride!=1, grouped, or dilated convolutions are skipped.

    Args:
        model:     nn.Module to modify (in-place).
        trainable: If True, use NovaWinogradConv2dTrainable (supports backward).
        verbose:   Print replacement summary.

    Returns:
        Number of layers replaced.
    """
    cls = NovaWinogradConv2dTrainable if trainable else NovaWinogradConv2d
    replaced = 0
    skipped = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Conv2d):
            continue
        if isinstance(module, (NovaWinogradConv2d, NovaWinogradConv2dTrainable)):
            continue

        ks = module.kernel_size
        stride = module.stride
        dilation = module.dilation
        groups = module.groups

        if ks != (3, 3) or stride != (1, 1) or dilation != (1, 1) or groups != 1:
            skipped += 1
            continue

        nova_conv = cls.from_conv2d(module)
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], nova_conv)
        replaced += 1

    if verbose:
        print(f"NOVA: replaced {replaced} Conv2d layers, skipped {skipped} ineligible")

    return replaced
