#!/usr/bin/env python3
"""
ResNet-50 inference with NOVA Winograd F(6,3).

Replaces all eligible 3x3 convolutions, compares output vs standard.
"""

import torch
import torchvision.models as models

from nova_winograd import replace_conv2d_with_nova


def main():
    device = torch.device("cuda")
    print("Loading ResNet-50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.eval()

    x = torch.randn(1, 3, 224, 224, device=device)

    # FP32 baseline
    with torch.no_grad():
        ref = model(x)

    # Convert to FP16 + NOVA
    model.half()
    n = replace_conv2d_with_nova(model)

    with torch.no_grad():
        nova_out = model(x.half())

    rel_err = (torch.norm(nova_out.float() - ref) / torch.norm(ref)).item()
    nan_count = torch.isnan(nova_out).sum().item()

    print(f"\nReplaced: {n} Conv2d layers")
    print(f"Rel error vs FP32: {rel_err:.4f}")
    print(f"NaN count: {nan_count}")
    print(f"Top-5 match: {ref.topk(5).indices.tolist()} vs {nova_out.topk(5).indices.tolist()}")


if __name__ == "__main__":
    main()
