"""
NOVA Winograd F(6,3) — Drop-in FP16 Winograd Convolution for AMD GPUs

Provides numerically stable F(6,3) Winograd convolution using NOVA's
optimal interpolation points. Runs on AMD Instinct GPUs via HIP + rocBLAS.

Quick start:
    from nova_winograd import NovaWinogradConv2d, replace_conv2d_with_nova

    # Option 1: Single layer replacement
    conv = NovaWinogradConv2d(64, 128, kernel_size=3, padding=1).cuda()
    output = conv(input_fp16)

    # Option 2: Replace all eligible layers in a model
    model = torchvision.models.resnet50(pretrained=True).cuda().half()
    replace_conv2d_with_nova(model)
    output = model(input_fp16)
"""

from nova_winograd.conv2d import (
    NovaWinogradConv2d,
    NovaWinogradConv2dTrainable,
    NovaWinogradConv2dCompilable,
)
from nova_winograd.ops import nova_forward, compute_tiling
from nova_winograd.surgery import replace_conv2d_with_nova

__version__ = "1.0.0"
__all__ = [
    "NovaWinogradConv2d",
    "NovaWinogradConv2dTrainable",
    "NovaWinogradConv2dCompilable",
    "nova_forward",
    "compute_tiling",
    "replace_conv2d_with_nova",
]
